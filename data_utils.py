import os

from sunpy.net import Fido, attrs as a
import sunpy
from sunpy import map
import astropy.io.fits as fits
from astropy.time import Time
import astropy.units as u

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interpolate


def get_coordinates(slits, path_to_slits, theta=0):
    """
    Get Coordinates
        Returns the HPC coordinates of a set of slits passed in



    :param slits:
        a list of strings, names of slits to get coordinates from
    :param path_to_slits:
        a string, the absolute path to the folder where the Hinode fits slits are stored
    :param theta:
        a float, roll angle correction to the p angle in the fits header (an angle offset to be minimized for later)

        The roll angle is the only offset parameter I can't correct for after the map is created,
        which is why it's passed in here. The other are corrected for later... might change this TODO: think about changing to feeding in all paramters here...

    :return coordinates:
        a numpy array, of shape (N_slits, 2, 192). HPC coordinates of each coordinate,
        (N_slits, x/y, position along slit)
    """

    coordinates_counter = 0
    coordinates = np.zeros((1, 2, 192))

    for slit in slits:
        temp_header = fits.open(path_to_slits + slit)[0].header

        temp_xcens = temp_header['XCEN']
        temp_ycens = temp_header['YCEN']
        temp_xdelt = temp_header['CDELT1']
        temp_ydelt = temp_header['CDELT2']

        temp_p1 = temp_header['CROTA1']
        temp_p2 = temp_header['CROTA1']  # TODO: This isn't being used... maybe a problem, check this...

        temp_coordinates = get_slit_coords(temp_xcens, temp_ycens, temp_xdelt, temp_ydelt, temp_p1, theta)

        if coordinates_counter == 0:
            coordinates[0] = temp_coordinates
        else:
            coordinates = np.vstack((coordinates, temp_coordinates))

        coordinates_counter += 1

    return coordinates


def get_slit_coords(xcen,
                    ycen,
                    xdelt,
                    ydelt,
                    p1,
                    theta=0):
    """
    Get Slit Coordinates: returns the HPC coordinates of a single HINODE slit, by reading in the header and
                          offseting it by the fitting parameters



    :param xcen:
        a float, total offset of the center of the dataset by x arcsec
    :param ycen:
        a float, total offset of the center of the dataset by y arcsec
    :param xdelt:
        a float, multiplier offset correction for spacing between pixels
    :param ydelt:
        a float, multiplier offset correction for spacing between pixels
    :param p1:
        roll angle, from FITs header
    :param theta:
        correction to roll angle from FITs header

    :return slit coordinates:
        a numpy array of shape
    """
    y_slit_indices = np.arange(-96, 96, 1)
    x_slit_indices = np.ones(192) * 0.5

    slit_coordinates_x = x_slit_indices * xdelt + xcen
    slit_coordinates_y = y_slit_indices * ydelt + ycen

    p = p1 + theta

    # roll corrected slit coords:
    slit_coordinates = np.zeros((1, 2, 192))
    slit_coordinates[0, 0] = slit_coordinates_x * np.cos(p * np.pi / 180) - slit_coordinates_y * np.sin(p * np.pi / 180)
    slit_coordinates[0, 1] = slit_coordinates_x * np.sin(p * np.pi / 180) + slit_coordinates_y * np.cos(p * np.pi / 180)

    return slit_coordinates


def interpolate(parameters,
                slits_sorted,
                full_hmi,
                hmix,
                hmiy,
                path_to_slits,
                slit_indices=(0, 192)):
   """
   Interpolate:
        for a subset of slits and offset parameters, interpolate HMI data onto
        the irregular grid of modified Hinode coordinates. Returns the map.



   :param parameters:
        a list of the 6 parameters which describe the offset of coordinates from the Hinode header
   :param slits_sorted: TODO: change this to just the subset of slits, not all slits and the indices
        a list of strings, the sorted names of all fits slit file names
   :param full_hmi:
        an (Nx, Ny) array of HMI magnetic, larger than the Hinode map (for padding issues),
        which will be interpolated from
   :param hmix:
        a corresponding (Nx, Ny) array of HP x coordinates in arcsec for each index

        a note: since HMI is rectangular, there are work-arounds, but this is what I found is the easiest way.
   :param hmiy:
        a corresponding (Nx, Ny) array of HP y coordinates in arcsec for each index
   :param path_to_slits:
        a string, the absolute path to the folder where the Hinode fits slits are stored
   :param slit_indices:

   :return: interpolated_HMI_B
        an array of the same size as the slits subset passed in. HMI magnetic data interpolated onto the Hinode
        grid from the Hinode header, offset by the input parameters.
   """
    dx = parameters[0] * u.arcsec
    dy = parameters[1] * u.arcsec

    deltax = parameters[2]
    deltay = parameters[3]

    theta = parameters[4]


    slits_subset = slits_sorted[slit_indices[0]:slit_indices[-1]]

    coordinates = get_coordinates(slits_subset, path_to_slits, theta)

    # unpacking coordinates into x and y arrays
    hinodex = coordinates[:, 1, :] * u.arcsec
    hinodey = coordinates[:, 0, :] * u.arcsec

    hinodex = hinodex * deltax - (1 - deltax) * hinodex[0, 0]
    hinodey = hinodey * deltay - (1 - deltay) * hinodey[0, 0]

    # pulling "best guess" corners of square HMI, size of Hinode raster, to define as low (original resolution) HMI data
    corner1_arcsec = (hinodex[0, 0] + dx, hinodey[0, 0] + dy)
    corner2_arcsec = (hinodex[-1, -1] + dx, hinodey[-1, -1] + dy)

    # pulling the closest HMI pixel to each corner, in x and y
    hmi_corner1_x_index = np.argmin(abs(corner1_arcsec[0] - hmix[0, :]))
    hmi_corner1_y_index = np.argmin(abs(corner1_arcsec[1] - hmiy[:, 0]))

    hmi_corner2_x_index = np.argmin(abs(corner2_arcsec[0] - hmix[0, :]))
    hmi_corner2_y_index = np.argmin(abs(corner2_arcsec[1] - hmiy[:, 0]))

    # expanding HMI box to avoid edge effects:
    delta = 10

    # pulling a regular, rectangular grid covering the HINODE raster from the corners above:
    hmi_x_coords = hmiy[hmi_corner2_x_index - delta:hmi_corner1_x_index + delta, hmi_corner2_y_index][
                   ::-1].value - dx.value
    hmi_y_coords = hmix[hmi_corner2_x_index, hmi_corner2_y_index - delta:hmi_corner1_y_index + delta][
                   ::-1].value - dy.value

    f = interpolate.RectBivariateSpline(hmi_y_coords, hmi_x_coords,
                                  full_hmi[hmi_corner2_y_index - delta:hmi_corner1_y_index + delta,
                                  hmi_corner2_x_index - delta:hmi_corner1_x_index + delta])

    # interpolating this square HMI data ONTO the irregular Hinode coordinates

    interpolated_HMI_B = f(hinodey, hinodex, grid=False)[:, ::-1]
    return interpolated_HMI_B


def fetch_data(path_to_slits,
               path_to_HMI = '~/sunpy/data/'):
    """
    Fetch Data:
        uses Sunpy's Fido to download all 45s HMI magnetograms covered by the given Hinode slits, +/- 1
        min on either side in case the closest HMI map to the first slit is before, and same for last slit for after)

        A known problem: may not work if Hinode observation crosses midnight (into a new day)

        This one uses a couple for loops, which isn't ideal for speed, but this only needs to be run once, and the big
        time use here is downloading the HMI data, so speed for reading the times of observation isn't a big time use.



    :param path_to_slits:
        Path to hinode fits slits. Must only be fits with headers in Hinode CSAC format - must be empty of other files

        Don't rename slits - the naming convention from CSAC means they can be sorted in order of observation time.

    :param path_to_HMI:
        Where to save fetched HMI magnetograms. Default is sunpy's default.

        Warning: this could be up to a few GB of data, depending on Hinode raster size.

    :return closest_index:
        An array of the index of the closest HMI 45s magnetogram to each slit.
        To be used for assembling upscaled artificial HMI image from multiple 45s magetograms.

        For instance, a closest_index == [0, 0, 0, 1, 1, 1, 2, 3, ...] means that the first 3 slits are closest to the
        0th (first in python indexing) HMI map, next three are closest to 1st (second in pyton indexing), etc...
    """

    # start by getting the observation times of the first and last slit, to know what HMI 45s magnetograms to download:
    slits = os.listdir(path_to_slits)
    slits_sorted = sorted(slits)  # sort in time via name - this may not be necessary, but just to make sure

    dateobs = []

    for slit in slits_sorted:
        temp_header = fits.open('./raster1_slits/' + slit)[0].header

        dateobs.append((temp_header['DATE_OBS']))

    # grab the first and last slits as start and end time
    starttime = dateobs[0]
    endtime = dateobs[-1]

    # turn the start and end time strings into astropy date-time objects, offset by 1 min on either side:
    starttime = Time(starttime) - 1*u.min
    endtime = Time(endtime) + 1*u.min

    # search using FIDO for all 45s HMI magnetograms in the time range:
    hmi_results = Fido.search(
        a.Time(starttime, endtime) & a.Instrument.hmi & a.Physobs('LOS_magnetic_field'))

    # this is for finding the closest (in time) HMI 45s magnetogram to each slit. Will be used later.
    closest_index = []

    for dateobs_i in dateobs:
        closest_index.append(np.argmin(abs((dateobs_i - hmi_results['vso']['Start Time']).value)))

    # fetch the data TODO: try to update this to only use data that's being used, don't download first/last if not needed
    path = path_to_HMI + '/{instrument}/align/'

    Fido.fetch(hmi_results[0], path = path)

    # TODO: somewhere in here, I need to read-in/save HMI coordinates... just need to do this once, since same for each HMI map.

    return closest_index

def read_in_HMI(path_to_HMI = '~/sunpy/data/'):
    """
    Read in HMI:
        Quick function to read in the locally saved HMI 45s data and stack it as a stacked numpy array,
        since I prefer working as numpy arrays.

        This may be possible by only using sunpy objects, but not in the way I implemented it.



    :param path_to_HMI:
        path to where HMI 45s data is stored. default is default sunpy save location (for Mac).

    :return: all_HMI_data
        A stacked numpy array of ALL HMI 45s magnetograms which are co-temporal with the Hinode raster.

        Shape is (4096, 4096, N_hmi_datasets)
    :return: hmix:
        array of HMI x coordinates, in arcsec, shape is (4069, 4069)
    :return: hmiy:
        array of HMI y coordinates, in arcsec, shape is (4069, 4069)
    """

    path = path_to_HMI + '/{instrument}/align/'

    all_HMI_maps = sorted(os.listdir(path))
    all_HMI_data = np.zeros((4096, 4096, 1))

    # save the first one, just to avoid stacking wrong with numpy concatenate:
    all_HMI_data[:, :, 0] = sunpy.map.Map(path + all_HMI_maps[0]).data
    hmix = sunpy.map.all_coordinates_from_map(path + all_HMI_maps[0]).Tx
    hmiy = sunpy.map.all_coordinates_from_map(path + all_HMI_maps[0]).Tx

    for i, HMI_map in enumerate(all_HMI_maps[1:]):
        if 'hmi_m_' in HMI_map:
            temp_data = np.reshape(sunpy.map.Map(path + HMI_map).data, (4096, 4096, 1))

            all_HMI_data = np.concatenate((all_HMI_data, temp_data), axis=2)
        else:
            pass

    return all_HMI_data, hmix, hmiy



def assemble_and_compare_interpolated_HMI(parameters,
                                          slits_sorted,
                                          path_to_slits,
                                          all_HMI_data,
                                          hmix,
                                          hmiy,
                                          hinode_B,
                                          closest_index,
                                          flag):
    """
    Assemble Interpolated HMI:
        Calls interpolate multiple times, for each HMI 45s magnetogram closest to at least one of the Hinode fits slits

        Finally, this compares the interpolated map to the inverted Hinode B, and produces a psuedo-chi squared.
        The minimize function then finds the parameter set to minimize the psuedo-chi squared.

        Was originally just going to assemble HMI, but in order to minimize parameters it's important that the assembly
        and the comparison are done in the same function.



    :param parameters:
        the set of 5 parameters to offset the Hinode dataset by
    :param slits_sorted:
        a list of strings, names of the slits in the Hinode observation
    :param path_to_slits:
        a string, absolute path to where the slits are saved
    :param all_HMI_data:
        the stacked array of all HMI data covering the observations
    :param hmix:
        hmi x coordinates, in arcsec
    :param hmiy:
        hmi y coordinates, in arcsec
    :param hinode_B:
        magnetic
    :param closest_index:
        a list of the index of the closest HMI dataset to each slit
    :param flag:
        true: return psuedo-chi-squared (for minimizing parameters)
        false: return assembled map (for vizualizing once minimization is done)

    :return interpolated_HMI:
        The final interpolated HMI image, created from multiple HMI 45s observations
    """

    N_slits = len(closest_index)
    interpolated_HMI = np.zeros((N_slits, 192))

    last_HMI_index = closest_index[-1]

    for i in range(last_HMI_index):
        # mask the closest_index array to only the ones closest to the current HMI dataset:
        slit_indices = np.array(np.where(np.array(closest_index) == i)[0])
        if slit_indices.size == 0:
            pass
        else:  # there are HMI maps corresponding to these slits
            index1 = slit_indices[0] - 1 # pad it by a row on either side
            index2 = slit_indices[-1] + 1
            interpolated_HMI[index1:index2, :][::-1, :] = interpolate(parameters,
                                                                        slits_sorted,
                                                                        all_HMI_data[:, :, i],
                                                                        hmix,
                                                                        hmiy,
                                                                        path_to_slits,
                                                                        (index1, index2))

    if flag:
        psuedo_chi_squared = np.sum((hinode_B[::-1] - interpolated_HMI) ** 2)
        return psuedo_chi_squared
    else:
        return interpolated_HMI


def minimize(initial_guess,
             slits_sorted,
             path_to_slits,
             all_HMI_data,
             hmix,
             hmiy,
             hinode_B,
             closest_index,
             bounds = None):
    """
    Minimize:
        a function which uses scipy's minimize to find the parameter set which best aligns the data



    :param initial_guess:
        a list of initial guess set of 5 parameters to start the minimizer at
    :param slits_sorted:
    :param path_to_slits:
    :param all_HMI_data:
    :param hmix:
    :param hmiy:
    :param hinode_B:
    :param closest_index:
    :param bounds:
        optional, a list of touples, used to bound the

    :return flag:
        a bool, if scipy's minimizer convereged
    :return parameters:
        if converged, return best fit paramters
    """

    # TODO: add functionality here to fix roll angle if fails to converge
    if bounds is not None:
        x = minimize(assemble_and_compare_interpolated_HMI,
                     x0=initial_guess,
                     method='Nelder-Mead',
                     args = (slits_sorted,
                             path_to_slits,
                             all_HMI_data,
                             hmix,
                             hmiy,
                             hinode_B,
                             closest_index,
                             True))
    else:
        x = minimize(assemble_and_compare_interpolated_HMI,
                     x0=initial_guess,
                     method='Nelder-Mead',
                     args=(slits_sorted,
                           path_to_slits,
                           all_HMI_data,
                           hmix,
                           hmiy,
                           hinode_B,
                           closest_index,
                           True),
                     bounds = bounds)

    return x.success, x.x

def plot_and_viz_compare():
    """
    Plot and visually compare:
        plot the interpolated HMI dataset vs. the inverted Hinode dataset, show the image.



    :return: None
    """

    plt.show()