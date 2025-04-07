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
from scipy.optimize import minimize as scipy_minimize
from scipy.ndimage import gaussian_filter as gf

import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Button, Slider
import matplotlib.patches as patches



def get_coordinates(slits,
                    i,
                    deltax,
                    deltay,
                    path_to_slits,
                    sizey,
                    theta=0):
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
        which is why it's passed in here. The other are corrected for later... might change this

    :return coordinates:
        a numpy array, of shape (N_slits, 2, sizey). HPC coordinates of each coordinate,
        (N_slits, x/y, position along slit)
    """
    coordinates_counter = 0
    coordinates = np.zeros((1, 2, sizey))

    for j, slit in enumerate(slits):
        temp_header = fits.open(path_to_slits + slit)[0].header

        temp_xcens = temp_header['XCEN']
        temp_ycens = temp_header['YCEN']
        temp_xdelt = temp_header['CDELT1']
        temp_ydelt = temp_header['CDELT2']

        temp_p1 = temp_header['CROTA1']

        index = i + j

        temp_coordinates = get_slit_coords(index,
                                           temp_xcens,
                                           temp_ycens,
                                           temp_xdelt,
                                           temp_ydelt,
                                           deltax,
                                           deltay,
                                           temp_p1,
                                           theta)

        if coordinates_counter == 0:
            coordinates[0] = temp_coordinates
        else:
            coordinates = np.vstack((coordinates, temp_coordinates))

        coordinates_counter += 1

    return coordinates


def get_slit_coords(index,
                    xcen,
                    ycen,
                    px,
                    py,
                    deltax,
                    deltay,
                    p1,
                    theta=0):
    """
    Get Slit Coordinates: returns the HPC coordinates of a single HINODE slit, by reading in the header and
                          offseting it by the fitting parameters



    :param index:
        a int, the index of the slit along the raster.
        Used for stretching the raster horizontally by the x-coordinate of each slit by xdelt * index
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
    x_slit_indices = np.ones(192) * 0.5
    y_slit_indices = np.arange(-96, 96, 1)

    slit_coordinates_x = x_slit_indices * px + xcen
    slit_coordinates_y = y_slit_indices * (py + deltay) + ycen

    p = p1 + theta

    # roll corrected slit coords:
    slit_coordinates = np.zeros((1, 2, 192))
    slit_coordinates[0, 0] = slit_coordinates_x * np.cos(p * np.pi / 180) - slit_coordinates_y * np.sin(p * np.pi / 180)
    slit_coordinates[0, 1] = slit_coordinates_x * np.sin(p * np.pi / 180) + slit_coordinates_y * np.cos(p * np.pi / 180)

    slit_coordinates[0, 0] = slit_coordinates[0, 0]
    slit_coordinates[0, 1] = slit_coordinates[0, 1] + index * deltax
    return slit_coordinates


def interpolate_section(parameters,
                        slits_sorted,
                        full_hmi,
                        hmix,
                        hmiy,
                        path_to_slits,
                        sizes,
                        slit_indices=(0, 192)):
    """
    Interpolate:
        for a subset of slits and offset parameters, interpolate HMI data onto
        the irregular grid of modified Hinode coordinates. Returns the map.



    :param parameters:
        a list of the 6 parameters which describe the offset of coordinates from the Hinode header
    :param slits_sorted:
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

    :param sizes:

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

    coordinates = get_coordinates(slits_subset,
                                  slit_indices[0],
                                  deltax,
                                  deltay,
                                  path_to_slits,
                                  sizes[1],
                                  theta)

    # unpacking coordinates into x and y arrays
    hinodex = coordinates[:, 0, :] * u.arcsec
    hinodey = coordinates[:, 1, :] * u.arcsec

    # pulling "best guess" corners of square HMI, size of Hinode raster, to define as low (original resolution) HMI data
    corner1_arcsec = (hinodex[0, 0] + dx, hinodey[0, 0] + dy)
    corner2_arcsec = (hinodex[-1, -1] + dx, hinodey[-1, -1] + dy)

    # pulling the closest HMI pixel to each corner, in x and y
    hmi_corner1_x_index = np.argmin(abs(corner1_arcsec[0] - hmix[0, :]))
    hmi_corner1_y_index = np.argmin(abs(corner1_arcsec[1] - hmiy[:, 0]))

    hmi_corner2_x_index = np.argmin(abs(corner2_arcsec[0] - hmix[0, :]))
    hmi_corner2_y_index = np.argmin(abs(corner2_arcsec[1] - hmiy[:, 0]))

    # expanding HMI box to avoid edge effects:
    delta = 30

    # pulling a regular, rectangular grid covering the HINODE raster from the corners above:
    hmi_x_coords = hmiy[hmi_corner2_x_index - delta:hmi_corner1_x_index + delta, hmi_corner2_y_index][
                   ::-1].value - dx.value
    hmi_y_coords = hmix[hmi_corner2_x_index, hmi_corner2_y_index - delta:hmi_corner1_y_index + delta][
                   ::-1].value - dy.value

    f = interpolate.RectBivariateSpline(hmi_y_coords,
                                        hmi_x_coords,
                                        full_hmi[hmi_corner2_y_index - delta:hmi_corner1_y_index + delta,
                                        hmi_corner2_x_index - delta:hmi_corner1_x_index + delta])

    # interpolating this square HMI data ONTO the irregular Hinode coordinates

    interpolated_HMI_B = f(hinodey, hinodex, grid=False)
    return interpolated_HMI_B


def fetch_data(path_to_slits,
               path_to_HMI='~/sunpy/data/',
               verbose=True):
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
        temp_header = fits.open(path_to_slits + slit)[0].header

        dateobs.append((temp_header['DATE_OBS']))

    # grab the first and last slits as start and end time
    starttime = dateobs[0]
    endtime = dateobs[-1]

    # turn the start and end time strings into astropy date-time objects, offset by 1 min on either side:
    starttime = Time(starttime) - 1 * u.min
    endtime = Time(endtime) + 1 * u.min

    # search using FIDO for all 45s HMI magnetograms in the time range:
    hmi_results = Fido.search(
        a.Time(starttime, endtime) & a.Instrument.hmi & a.Physobs('LOS_magnetic_field'))

    # this is for finding the closest (in time) HMI 45s magnetogram to each slit. Will be used later.
    closest_index = []

    for dateobs_i in dateobs:
        dateobs_i = Time(dateobs_i)
        closest_index.append(np.argmin(abs((dateobs_i - hmi_results['vso']['Start Time']).value)))

    # fetch the data
    path = path_to_HMI + '/data/{instrument}/align/'

    Fido.fetch(hmi_results[0], path=path, progress=verbose)

    return closest_index


def read_in_HMI(path_to_HMI='/Users/jamescrowley/sunpy/'):
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

    path = path_to_HMI + '/data/HMI/align/'

    all_HMI_maps = sorted(os.listdir(path))
    all_HMI_data = np.zeros((4096, 4096, 1))

    # save the first one, just to avoid stacking wrong with numpy concatenate:
    test_map = sunpy.map.Map(path + all_HMI_maps[0])

    all_HMI_data[:, :, 0] = sunpy.map.Map(path + all_HMI_maps[0]).data
    hmix = sunpy.map.all_coordinates_from_map(test_map).Tx
    hmiy = sunpy.map.all_coordinates_from_map(test_map).Ty

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
                                          sizes,
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
    :param sizes:

    :param flag:
        true: return pseudo-chi-squared (for minimizing parameters)
        false: return assembled map (for visualizing once minimization is done)

    :return interpolated_HMI:
        The final interpolated HMI image, created from multiple HMI 45s observations
    """

    interpolated_HMI = np.zeros((sizes[0], sizes[1]))

    last_HMI_index = closest_index[-1]

    for i in range(last_HMI_index + 1):
        # mask the closest_index array to only the ones closest to the current HMI dataset:
        slit_indices = np.array(np.where(np.array(closest_index) == i)[0])
        if slit_indices.size == 0:
            pass
        else:  # there are HMI maps corresponding to these slits
            index1 = slit_indices[0]
            index2 = slit_indices[-1] + 1  # pad it by a row, this makes it the right size
            try:
                interpolated_HMI[:, index1:index2][::-1, ::-1] = interpolate_section(parameters,
                                                                                     slits_sorted,
                                                                                     all_HMI_data[:, :, i],
                                                                                     hmix,
                                                                                     hmiy,
                                                                                     path_to_slits,
                                                                                     sizes,
                                                                                     (index1, index2)).T
            except:
                index2 -= 1
                interpolated_HMI[:, index1:index2][::-1, ::-1] = interpolate_section(parameters,
                                                                                     slits_sorted,
                                                                                     all_HMI_data[:, :, i],
                                                                                     hmix,
                                                                                     hmiy,
                                                                                     path_to_slits,
                                                                                     sizes,
                                                                                     (index1, index2)).T
    if flag:
        S0 = np.zeros_like(hinode_B)
        S0[abs(hinode_B) > 80] = 1

        S1 = gf(S0, sigma=0.7)

        S2 = np.zeros_like(S1)
        S2[S1 > 0.7] = 1

        Q = np.sum(abs(interpolated_HMI) * S2)

        return 1 / Q

    else:
        return interpolated_HMI


def plot_and_viz_compare(hinode_B,
                         HMI_B):
    """
    Plot and visually compare:
        plot the interpolated HMI dataset vs. the inverted Hinode dataset, show the image.



    :return: None
    """

    fps = 30

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(8, 8))

    # stacking the data on-top:
    B_data = [hinode_B[::-1, :][:-3], HMI_B[::-1, :][:-3], hinode_B[::-1, :][:-3]] * 100

    ax = plt.subplot()
    im = ax.imshow(B_data[0], cmap='PuOr', vmin=-np.std(B_data[0]), vmax=np.std(B_data[0]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    def animate_func(i):
        if i % fps == 0:
            print('.', end='')

        im.set_array(B_data[i])

    anim = animation.FuncAnimation(fig,
                                   animate_func,
                                   frames=len(B_data),
                                   interval=15000 / fps)  # in ms
    plt.show()


def minimize(initial_guess,
             slits_sorted,
             path_to_slits,
             all_HMI_data,
             hmix,
             hmiy,
             hinode_B,
             closest_index,
             sizes,
             bounds=None):
    """
    Minimize:
        a function which uses scipy's minimize to find the parameter set which best aligns the data

    :param initial_guess:
        a list of initial guess set of 5 parameters to start the minimizer at
    :param slits_sorted:
        a list of strings, the alphabetically (and therefore temporally) sorted fits slits in the path_to_slits folder
    :param path_to_slits:
        a string, path to the folder where the unpacked CSAC slits are saved.
    :param all_HMI_data:
        The stacked numpy of all co-temporal hmi signed magnetic field data.
        Shape should be (4096, 4096, t), where t is the number of co-temporal datasets.
    :param hmix:
        (4096 x 4096) array of hmix coordinates, the same for every hmi 45s map.
    :param hmiy:
        (4096 x 4096) array of hmiy coordinates, the same for every hmi 45s map.
    :param hinode_B:
        a numpy array of signed hinode magnetic field to be aligned. Should be (Nx, Ny) in shape.
    :param closest_index:
        a list of ints, the index of the closest hmi co-temporal map to each slit. i.e. a list of N_slits ints.
    :param bounds:
        a list of tuples, used to bound the parameter search
    :param sizes:

    :return flag:
        a bool, if scipy's minimize convereged
    :return parameters:
        if converged, return best fit paramters
    """

    if bounds is not None:
        x = scipy_minimize(assemble_and_compare_interpolated_HMI,
                           x0=initial_guess,
                           method='Nelder-Mead',
                           args=(slits_sorted,
                                 path_to_slits,
                                 all_HMI_data,
                                 hmix,
                                 hmiy,
                                 hinode_B,
                                 closest_index,
                                 sizes,
                                 True))
    else:
        x = scipy_minimize(assemble_and_compare_interpolated_HMI,
                           x0=initial_guess,
                           method='Nelder-Mead',
                           args=(slits_sorted,
                                 path_to_slits,
                                 all_HMI_data,
                                 hmix,
                                 hmiy,
                                 hinode_B,
                                 closest_index,
                                 sizes,
                                 True),
                           bounds=bounds)

    return x.success, x.x


def run(path_to_slits,
        hinode_B,
        p0=None,
        bounds=None,
        path_to_sunpy='/Users/jamescrowley/sunpy/',
        plot=True,
        save_coords=False,
        save_params=False,
        verbose=True):
    """
    Run:
        I want this to be the funciton which calls all the others to run the alignment, almost like a main

    :param path_to_slits:
        a string, path to the folder where the unpacked CSAC slits are saved.

    :param hinode_B:
        a numpy array of signed hinode magnetic field to be aligned. Should be (Nx, Ny) in shape.

    :param p0:
        initial guess for parameters.

    :param bounds:
        a list of floats, initial bounds around which to search for parameters. Defualts to none and use wide bounds.
        Update if you want to use tighter bounds in parameter search.

    :param path_to_sunpy:
        a string, path to where sunpy data is to be saved.

    :param plot:
        bool, whether to plot the output image. default is .

    :param save_coords:
        bool, whether to save HPC coords. Saves as a fits file wherever script is run, format is
        [i_index, j_index, 2], i.e. [:, :, 0] is HPC x and [:, :, 1] is HPC y.

    :param save_params:
        bool, whether to save fitted parameters. Saves as a fits file wherever script is run, format is
        [x_cen, y_cen, p_x, p_y, theta].

    :param verbose:
        bool, whether to output progress messages to console. Defualt is true.



    :return: 
    """

    # read names of slits:
    slits = os.listdir(path_to_slits)
    slits_sorted = sorted(slits)  # sort in time via name - this may not be necessary, but just to make sure
    N_slits = len(slits_sorted)

    sizex = fits.open(path_to_slits + '/' + sorted(os.listdir(path_to_slits))[-1])[0].header['SLITINDX'] + 1
    sizey = fits.open(path_to_slits + '/' + sorted(os.listdir(path_to_slits))[-1])[0].data.shape[1]

    sizes = (sizex, sizey)

    if verbose:
        print('Fits slits read in. Number of slits: ' + str(N_slits))
        print('Size of dataset: ' + str(sizey) + ' x ' + str(sizex))

    if hinode_B is None:
        hinode_B = create_psuedo_B(sizes, path_to_slits, slits_sorted)

    # download and read-in all the needed HMI data:
    closest_index = fetch_data(path_to_slits, path_to_sunpy, verbose)
    all_HMI_data, hmix, hmiy = read_in_HMI()

    if verbose:
        print('Fido successfully downloaded HMI data.')

        print(50 * '-')
        print('Performing Initial Rough Alignment')

    if p0 is None:
        p0 = [14, 34, -8.52e-03, 3.72e-04, 2.5]
    closest_index0 = N_slits * [1]
    if bounds is None:
        bounds = [(-40, 40), (-40, 40), (0.9, 1.1), (0.9, 1.1), (2, 4)]
    else:
        bounds = bounds

    converged, parameters = minimize(p0,
                                     slits_sorted,
                                     path_to_slits,
                                     all_HMI_data,
                                     hmix,
                                     hmiy,
                                     hinode_B,
                                     closest_index0,
                                     sizes,
                                     bounds)
    if verbose:
        print('Initial Rough Alignment Complete.')
        print('Estimate of parameters: ' + str(parameters))
        print(50 * '-')
        print('Performing Final Fit')

    converged, parameters = minimize(parameters,
                                     slits_sorted,
                                     path_to_slits,
                                     all_HMI_data,
                                     hmix,
                                     hmiy,
                                     hinode_B,
                                     closest_index,
                                     sizes,
                                     bounds)

    if verbose:
        print('Minimized: ' + str(converged))
        print('Final Parameters: ' + str(parameters))

    if not converged:
        raise Exception('Failed to Solve. Exiting.')

    if plot:
        # after converged, vizualize it:
        if verbose:
            print(50 * '-')
            print('Vizualizing Final Solution:')
        final_HMI = assemble_and_compare_interpolated_HMI(parameters,
                                                          slits_sorted,
                                                          path_to_slits,
                                                          all_HMI_data,
                                                          hmix,
                                                          hmiy,
                                                          hinode_B,
                                                          closest_index,
                                                          sizes,
                                                          False)

        plot_and_viz_compare(hinode_B, final_HMI)

    all_HMI_files = os.listdir(path_to_sunpy + '/data/HMI/align/')

    # remove the HMI maps
    for file in all_HMI_files:
        os.remove(path_to_sunpy + '/data/HMI/align/' + file)

    # save the coordinate, and any other requested information
    if save_coords:  # if saving, create the final coordinate arrays.

        dx = (parameters[0]) * u.arcsec
        dy = (parameters[1]) * u.arcsec

        deltax = parameters[2]
        deltay = parameters[3]

        theta = parameters[4]

        final_coordinates = get_coordinates(slits=slits,
                                            i=0,
                                            deltax=deltax,
                                            deltay=deltay,
                                            path_to_slits=path_to_slits,
                                            sizey=sizey,
                                            theta=theta)

        finalx = final_coordinates[:, 0, :]
        finaly = final_coordinates[:, 1, :]

        finalx = (finalx * (1 - deltax)) + dx.value
        finaly = (finaly * (1 - deltay)) + dy.value

        Nx = hinode_B.shape[0]
        Ny = hinode_B.shape[1]

        output = np.zeros((finalx.shape[0], finalx.shape[1], 2))

        output[:Nx, :Ny, 0] = finalx[:Nx, :Ny]
        output[:Nx, :Ny, 1] = finaly[:Nx, :Ny]

        output = np.sort(output, axis=0)  # don't know if this matters but to be safe, sort coords / indices together

        fits.writeto('coordinates.fits', output, overwrite=True)
    if save_params:
        fits.writeto('parameters.fits', parameters, overwrite=True)


def create_psuedo_B(sizes,
                    path_to_slits,
                    slits_sorted,
                    std_B=50):
    """
    If you don't have an inverted magnetic field, create a psuedo-B from a sum over signed circular polarization of
    second 6302 wing - works fairly well for QS datasets, need to check for active datasets.

    :param sizes:
    :param path_to_slits:
    :param slits_sorted:
    :param std_B:

    :return:
    """

    psuedo_B = np.zeros((sizes[0], sizes[1]))

    for i, slit in enumerate(slits_sorted):
        slit_temp = fits.open(path_to_slits + slit)[0].data
        psuedo_B[:, i] = -np.sum(slit_temp[3, :, -35:], axis=1)

    psuedo_B = psuedo_B / np.std(psuedo_B) * std_B

    return psuedo_B


def show_gui(p0,
             HMI_data,
             initial_width=500
             ):
    """
    Code to show a small GUI showing the initial rough alignment. Values from sliders will
    :param parameters:
    :return:
    """

    initial_x0, initial_y0, initial_deltax, initial_deltay, initial_theta = p0

    fig, axd = plt.subplot_mosaic(
        """
        AAAA.CC
        BBBB.CC
        """)

    sub1 = axd['A']
    sub2 = axd['B']
    sub3 = axd['C']

    rect = patches.Rectangle((2048 - initial_width / 2, 2048 - initial_width / 2), initial_width, initial_width,
                             linewidth=1, edgecolor='r', facecolor='none', ls='--')

    ### Reset Button:
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    axamp = fig.add_axes([0.6, 0.15, 0.3, 0.03])
    range_slider = Slider(
        ax=axamp,
        label="Range",
        valmin=1,
        valmax=4096,
        valinit=initial_width,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.45, 0.3, 0.03])
    x0_slider = Slider(
        ax=axamp,
        label=r"$x_0$",
        valmin=0,
        valmax=4000,
        valinit=initial_x0,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.40, 0.3, 0.03])
    y0_slider = Slider(
        ax=axamp,
        label=r"$y_0$",
        valmin=0,
        valmax=4000,
        valinit=initial_y0,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.35, 0.3, 0.03])
    deltax_slider = Slider(
        ax=axamp,
        label=r"$\delta_x$",
        valmin=0,
        valmax=500,
        valinit=initial_deltax,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.30, 0.3, 0.03])
    deltay_slider = Slider(
        ax=axamp,
        label=r"$\delta_y$",
        valmin=0,
        valmax=500,
        valinit=initial_deltay,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.25, 0.3, 0.03])
    theta_slider = Slider(
        ax=axamp,
        label=r"$\theta$",
        valmin=0,
        valmax=500,
        valinit=initial_deltay,
        orientation="horizontal"
    )

    data = HMI_data

    # WAY downsampling context image to make plotting faster:
    sub1.imshow(data[::10, ::10],
                vmin=-100,
                vmax=100,
                cmap='gray',
                origin='lower',
                extent=[0, data.shape[0], 0, data.shape[1]])
    scatter1 = sub1.plot(x0_slider.val, y0_slider.val, marker='x', ms=10, c='r')[0]
    scatter2 = sub2.plot(x0_slider.val, y0_slider.val, marker='x', ms=10, c='r')[0]
    sub1.add_patch(rect)

    sub2.imshow(data[:, :], vmin=-100, vmax=100, cmap='gray', origin='lower')
    sub2.set_xlim(2048 - 500, 2048 + 500)
    sub2.set_ylim(2048 - 500, 2048 + 500)
    sub3.axis('off')
    def update_hmi_frame(val):
        sub2.set_xlim((x0_slider.val - range_slider.val / 2, x0_slider.val + range_slider.val / 2))
        sub2.set_ylim((y0_slider.val - range_slider.val / 2, y0_slider.val + range_slider.val / 2))
        rect.set_width(range_slider.val)
        rect.set_height(range_slider.val)

        rect.set_x(x0_slider.val - range_slider.val / 2)
        rect.set_y(y0_slider.val - range_slider.val / 2)

        # keeping this commmented, I think this re-renders, i just want to change display things without re-rendering for time
        # fig.canvas.draw_idle()

    range_slider.on_changed(update_hmi_frame)
    x0_slider.on_changed(update_hmi_frame)
    y0_slider.on_changed(update_hmi_frame)
    def update_hinode_frame(val):
        scatter1.set_data(x0_slider.val, y0_slider.val)
        scatter2.set_data(x0_slider.val, y0_slider.val)

    x0_slider.on_changed(update_hinode_frame)
    y0_slider.on_changed(update_hinode_frame)
    deltax_slider.on_changed(update_hinode_frame)
    deltay_slider.on_changed(update_hinode_frame)
    theta_slider.on_changed(update_hinode_frame)

    def reset(event):
        range_slider.reset()

        x0_slider.reset()
        y0_slider.reset()
        deltax_slider.reset()
        deltay_slider.reset()
        theta_slider.reset()

    button.on_clicked(reset)

    plt.show()
