import os

from sunpy.net import Fido, attrs as a
import sunpy
from sunpy import map
import astropy.io.fits as fits
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord

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
                    deltax,
                    deltay,
                    path_to_slits,
                    sizey,
                    theta=0):
    """
    Get Coordinates
        Returns the HPC coordinates of a set of slits passed in

    # TODO: fill these docustrings in:

    :param slits:
        a list of strings, names of slits to get coordinates from
    :param deltax:
    :param deltay:
    :param path_to_slits:
        a string, the absolute path to the folder where the Hinode fits slits are stored
    :param sizey:
        an int, number of pixels along slit
    :param theta:
        a float, roll angle correction to the p angle in the fits header (an angle offset to be minimized for later)

    :return coordinates:
        a numpy array, of shape (N_slits, 2, sizey). HPC coordinates of each coordinate,
        (N_slits, x/y, position along slit)
    """
    sizex = len(slits)

    coordinates = np.zeros((sizex, sizey, 2))

    first_header = fits.open(path_to_slits + slits[0])[0].header
    px = first_header['CDELT2']
    p1 = first_header['CROTA1']

    p = p1 + theta

    for j, slit in enumerate(slits):
        temp_header = fits.open(path_to_slits + slit)[0].header
        index = temp_header['SLITINDX']

        temp_coordinates = get_slit_coordinates(index,
                                                px,
                                                deltax,
                                                deltay,
                                                p)

        coordinates[j] = temp_coordinates

    return coordinates


def get_slit_coordinates(index,
                         px,
                         deltax,
                         deltay,
                         p=0):
    """
    Get Slit Coordinates: returns the HPC coordinates of a single HINODE slit specified by the header, can be stretched 
    and rolled by fitting parametrs  

    :param sizey:
        int, number of pixels along slit
    :param index:
        a int, the index of the slit along the raster.
        Used for determining coordinate of raster through coordinate transformation
    :param px:
        initial plate scale in x, nominally from header
    :param deltax:
        modification to plate scale in x, to be fitted for
    :param deltay:
        modification to plate scale in y, to be fitted for
    :param p:
        roll angle, including both angle from header and theta offset angle  
        
    :return slit coordinates:
        a numpy array of shape
    """
    x_slit_indices = np.ones(192) * 0.5
    y_slit_indices = np.arange(-96, 96, 1) * (px + deltay)

    # roll slit by p:
    slit_coordinates = np.zeros((192, 2))
    slit_coordinates[:, 0] = x_slit_indices * np.cos(p * np.pi / 180) - y_slit_indices * np.sin(p * np.pi / 180)
    slit_coordinates[:, 1] = x_slit_indices * np.sin(p * np.pi / 180) + y_slit_indices * np.cos(p * np.pi / 180)

    # correct the roll in each slit for the offset from the first slit:
    offset_from_first_slit = index * (deltax + px)

    x_offset_from_first_slit = offset_from_first_slit * np.cos(p * np.pi / 180)
    y_offset_from_first_slit = offset_from_first_slit * np.sin(p * np.pi / 180)

    slit_coordinates[:, 0] += x_offset_from_first_slit
    slit_coordinates[:, 1] += y_offset_from_first_slit

    return slit_coordinates


def create_image_interpolator(image, **interpolator_args):
    """
    Create a RegularGridInterpolator that will take floating point pixel coordinates and interpolate
    the image array. Here we assume the pixel coordinates are indexed the same as the array, which
    is zero-indexed, C-Style Y,X ordering.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
    for the interpolator keyword arguments. The important ones are:
     - method: interpolation style (nearest, linear, cubic, etc.). nearest is good for showing actual pixel values.
     - bounds_error: set this to "False" if your image only partially covers the destination map grid.
     - fill_value: set this value to fill in pixels that are out of bounds.
    """
    # get the dimensions
    nx = image.shape[1]
    ny = image.shape[0]

    # the 1D scales here are just the pixel indexes
    pix_x1d = np.arange(nx, dtype=np.float64)
    pix_y1d = np.arange(ny, dtype=np.float64)

    # build the interpolator
    interpolator = interpolate.RegularGridInterpolator(
        (pix_y1d, pix_x1d), image, **interpolator_args)

    return interpolator


def interpolate_sunpy(coords,
                      full_hmi,
                      full_hmi_wcs):
    """
        Interpolate:
            for a subset of slits and offset parameters, interpolate HMI data onto
            the irregular grid of modified Hinode coordinates. Returns the map.


            In this one, I want to pass in coordinates, not parameters (keep it to just the interpolation)


    """

    interpolator = create_image_interpolator(full_hmi)

    pixel_coords = full_hmi_wcs.world_to_pixel(coords)
    pix_x = pixel_coords[0]
    pix_y = pixel_coords[1]
    points = np.stack([pix_y, pix_x], axis=len(pix_x.shape))
    values = interpolator(points)

    return values


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



    :param verbose: bool, to print progress messages
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

        dateobs.append(Time(temp_header['DATE_OBS']))

    # grab the first and last slits as start and end time
    starttime = dateobs[0]
    endtime = dateobs[-1]

    # turn the start and end time strings into astropy date-time objects, offset by 1 min on either side:
    starttime = (starttime) - 1 * u.min
    endtime = (endtime) + 1 * u.min

    # search using FIDO for all 45s HMI magnetograms in the time range:
    hmi_results = Fido.search(
        a.Time(starttime, endtime) & a.Instrument.hmi & a.Physobs('LOS_magnetic_field'))

    # this is for finding the closest (in time) HMI 45s magnetogram to each slit. Will be used later.
    closest_index = []

    for dateobs_i in dateobs:
        closest_index.append(np.argmin(abs((dateobs_i - hmi_results['vso']['Start Time']).value)))

    # fetch the data
    path = path_to_HMI + '/data/{instrument}/align/'

    Fido.fetch(hmi_results[0], path=path, progress=verbose)

    return closest_index, dateobs


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

    return all_HMI_data, hmix, hmiy, test_map.wcs


def assemble_and_compare_interpolated_HMI(parameters,
                                          slits_sorted,
                                          path_to_slits,
                                          all_HMI_data,
                                          hmi_wcs,
                                          hinode_B,
                                          closest_index,
                                          sizes,
                                          obstime,
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
                coords_section = get_coordinates(slits_sorted[index1:index2],
                                                 parameters[2],
                                                 parameters[3],
                                                 path_to_slits,
                                                 sizes[1],
                                                 parameters[4])

                coords_section[..., 0] += parameters[0]
                coords_section[..., 1] += parameters[1]
                coords_section = np.sort(coords_section, axis=0)

                hmi_image = all_HMI_data[..., i]
                hmi_wcs = hmi_wcs

                mean_time = Time(obstime[index1:index2]).mean()
                earth_pos = sunpy.coordinates.get_earth(mean_time)
                frame_hpc = sunpy.coordinates.Helioprojective(obstime=mean_time, observer=earth_pos)
                coord_hpc = SkyCoord(coords_section[..., 0] * u.arcsec,
                                     coords_section[..., 1] * u.arcsec,
                                     frame=frame_hpc)

                interpolated_HMI[:, index1:index2][::-1, :] = interpolate_sunpy(coord_hpc,
                                                                                   hmi_image,
                                                                                   hmi_wcs).T
            except:
                index2 -= 1
                coords_section = get_coordinates(slits_sorted[index1:index2],
                                                 parameters[2],
                                                 parameters[3],
                                                 path_to_slits,
                                                 sizes[1],
                                                 parameters[4])

                coords_section[..., 0] += parameters[0]
                coords_section[..., 1] += parameters[1]
                coords_section = np.sort(coords_section, axis=0)

                hmi_image = all_HMI_data[..., i]
                hmi_wcs = hmi_wcs

                mean_time = Time(obstime[index1:index2]).mean()
                earth_pos = sunpy.coordinates.get_earth(mean_time)
                frame_hpc = sunpy.coordinates.Helioprojective(obstime=mean_time, observer=earth_pos)
                coord_hpc = SkyCoord(coords_section[..., 0] * u.arcsec,
                                     coords_section[..., 1] * u.arcsec,
                                     frame=frame_hpc)

                interpolated_HMI[:, index1:index2][::-1, :] = interpolate_sunpy(coord_hpc,
                                                                                   hmi_image,
                                                                                   hmi_wcs).T
    if flag:
        S0 = np.zeros_like(hinode_B)
        S0[abs(hinode_B) > 80] = 1

        S1 = gf(S0, sigma=0.8)

        S2 = np.zeros_like(S1)
        S2[S1 > 0.7] = 1

        Q = np.sum(abs(interpolated_HMI[::-1]) * S2)

        return 1 / Q

    else:
        return interpolated_HMI[::-1]


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
             hmi_wcs,
             hinode_B,
             closest_index,
             sizes,
             obstime,
             bounds=None):
    """
    Minimize:
        a function which uses scipy's minimize to find the parameter set which best aligns the data

    :param obstime:
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
    :param sizes:

    :param bounds:
        a list of tuples, used to bound the parameter search


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
                                 hmi_wcs,
                                 hinode_B,
                                 closest_index,
                                 sizes,
                                 obstime,
                                 True))
    else:
        x = scipy_minimize(assemble_and_compare_interpolated_HMI,
                           x0=initial_guess,
                           method='Nelder-Mead',
                           args=(slits_sorted,
                                 path_to_slits,
                                 all_HMI_data,
                                 hmi_wcs,
                                 hinode_B,
                                 closest_index,
                                 sizes,
                                 obstime,
                                 True),
                           bounds=bounds,
                           tol=0.00001)

    return x.success, x.x


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


def show_gui(parameters,
             HMI_data,
             hmix,
             hmiy,
             coords,
             hinode_B
             ):
    """
    Code to show a small GUI showing the initial rough alignment. Values from sliders will
    :param HMI_data: numpy array of single HMI data in fits format to plot as background.
    :param hmix: numpy array of HPC coordinates corresponding to HMI x coords
    :param hmiy: numpy array of HPC coordinates corresponding to HMI x coords
    :param coords: numpy array of Hinode coords
    :param hinode_B: numpy arary of hinode magnetic field in shape of coords to plot
    :param parameters: initial parameters to set slider initial values

    :return: parameters:
        a list of parameters, returned when done button in gui is pressed
    """
    initial_width = 100
    initial_x0, initial_y0, initial_deltax, initial_deltay, initial_theta = parameters
    data = HMI_data

    middlex = np.mean(coords[:, :, 0])
    middley = np.mean(coords[:, :, 1])

    fig, axd = plt.subplot_mosaic(
        """
        AAAA.CC
        BBBB.CC
        """,
        figsize=(12, 8))

    sub1 = axd['A']
    sub2 = axd['B']
    sub3 = axd['C']

    # Reset Button:
    resetax = fig.add_axes([0.6, 0.025, 0.1, 0.04])
    button_reset = Button(resetax, 'Reset', hovercolor='0.975')

    # Done Button:
    doneax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button_done = Button(doneax, 'Done', hovercolor='0.975')

    axamp = fig.add_axes([0.6, 0.15, 0.3, 0.03])
    range_slider = Slider(
        ax=axamp,
        label="Range",
        valmin=1,
        valmax=400,
        valinit=initial_width,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.45, 0.3, 0.03])
    x0_slider = Slider(
        ax=axamp,
        label=r"$x_0$",
        valmin=initial_x0 - 100,
        valmax=initial_x0 + 100,
        valinit=initial_x0,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.40, 0.3, 0.03])
    y0_slider = Slider(
        ax=axamp,
        label=r"$y_0$",
        valmin=initial_y0 - 100,
        valmax=initial_y0 + 100,
        valinit=initial_y0,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.35, 0.3, 0.03])
    deltax_slider = Slider(
        ax=axamp,
        label=r"$\delta_x$",
        valmin=initial_deltax - 0.1,
        valmax=initial_deltax + 0.1,
        valinit=initial_deltax,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.30, 0.3, 0.03])
    deltay_slider = Slider(
        ax=axamp,
        label=r"$\delta_y$",
        valmin=initial_deltay - 0.1,
        valmax=initial_deltay + 0.1,
        valinit=initial_deltay,
        orientation="horizontal"
    )

    axamp = fig.add_axes([0.6, 0.25, 0.3, 0.03])
    theta_slider = Slider(
        ax=axamp,
        label=r"$\theta$",
        valmin=initial_theta - 10,
        valmax=initial_theta + 10,
        valinit=initial_theta,
        orientation="horizontal",
        track_color='r',
    )

    axamp = fig.add_axes([0.15, 0.6, 0.01, 0.3])
    hmi_clim_slider = Slider(
        ax=axamp,
        label="hmi clim",
        valmin=0,
        valmax=300,
        valinit=100,
        orientation="vertical"
    )

    axamp = fig.add_axes([0.15, 0.15, 0.01, 0.3])
    sp_clim_slider = Slider(
        ax=axamp,
        label="SP clim",
        valmin=0,
        valmax=300,
        valinit=100,
        orientation="vertical"
    )

    # WAY down-sampling context image to make plotting faster:
    hmi_image_full = sub1.imshow(data[::15, ::15][::-1, ::-1],
                                 vmin=-100,
                                 vmax=100,
                                 cmap='gray',
                                 origin='lower',
                                 extent=[hmix[-1, -1], hmix[0, 0], hmiy[-1, -1], hmiy[0, 0]])
    sub2.set_ylim(hmix[-1, -1], hmix[0, 0])
    sub2.set_xlim(hmiy[-1, -1], hmiy[0, 0])

    image1 = sub1.plot(middlex + x0_slider.val - initial_x0,
                       middley + y0_slider.val - initial_y0, marker='x', ms=10, c='r')[0]

    rect = patches.Rectangle((middlex - initial_width / 2, middley - initial_width / 2),
                             initial_width,
                             initial_width,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none',
                             ls='--')

    sub1.add_patch(rect)

    hmi_image_subset = sub2.imshow(data[:, :][::-1, ::-1],
                                   vmin=-50,
                                   vmax=50,
                                   cmap='gray',
                                   origin='lower',
                                   extent=[hmix[-1, -1], hmix[0, 0], hmiy[-1, -1], hmiy[0, 0]])
    sub2.set_xlim(middlex - initial_width / 2, middlex + initial_width / 2)
    sub2.set_ylim(middley - initial_width / 2, middley + initial_width / 2)

    image2 = sub2.imshow(hinode_B,
                         vmin=-50,
                         vmax=50,
                         cmap='PuOr',
                         alpha=0.5,
                         extent=[coords[0, 0, 0], coords[-1, -1, 0], coords[0, 0, 1], coords[-1, -1, 1]],
                         origin='lower')

    sub3.axis('off')

    def update_hmi_frame(val):
        sub2.set_xlim((middlex + x0_slider.val - initial_x0 - range_slider.val / 2,
                       middlex + x0_slider.val - initial_x0 + range_slider.val / 2))
        sub2.set_ylim((middley + y0_slider.val - initial_y0 - range_slider.val / 2,
                       middley + y0_slider.val - initial_y0 + range_slider.val / 2))
        rect.set_width(range_slider.val)
        rect.set_height(range_slider.val)

        rect.set_x(middlex + x0_slider.val - initial_x0 - range_slider.val / 2)
        rect.set_y(middley + y0_slider.val - initial_y0 - range_slider.val / 2)

        fig.canvas.draw_idle()
        hmi_image_full.set_clim((-hmi_clim_slider.val / 2, hmi_clim_slider.val / 2))

    range_slider.on_changed(update_hmi_frame)
    x0_slider.on_changed(update_hmi_frame)
    y0_slider.on_changed(update_hmi_frame)

    hmi_clim_slider.on_changed(update_hmi_frame)

    def update_hinode_frame(val):
        image1.set_data(middlex + x0_slider.val - initial_x0,
                        middley + y0_slider.val - initial_y0)

        width_x = coords[-1, -1, 0] - coords[0, 0, 0]
        width_y = coords[-1, -1, 1] - coords[0, 0, 1]

        image2.set_extent((coords[0, 0, 0] + (x0_slider.val - initial_x0) - width_x * deltax_slider.val,
                           coords[-1, -1, 0] + (x0_slider.val - initial_x0) + width_x * deltax_slider.val,
                           coords[0, 0, 1] + (y0_slider.val - initial_y0) - width_y * deltay_slider.val,
                           coords[-1, -1, 1] + (y0_slider.val - initial_y0) + width_y * deltay_slider.val))

        hmi_image_subset.set_clim((-hmi_clim_slider.val / 2, hmi_clim_slider.val / 2))
        image2.set_clim((-sp_clim_slider.val / 2, sp_clim_slider.val / 2))

    x0_slider.on_changed(update_hinode_frame)
    y0_slider.on_changed(update_hinode_frame)
    deltax_slider.on_changed(update_hinode_frame)
    deltay_slider.on_changed(update_hinode_frame)
    theta_slider.on_changed(update_hinode_frame)

    hmi_clim_slider.on_changed(update_hinode_frame)
    sp_clim_slider.on_changed(update_hinode_frame)

    def reset(event):
        range_slider.reset()

        x0_slider.reset()
        y0_slider.reset()
        deltax_slider.reset()
        deltay_slider.reset()
        theta_slider.reset()
        hmi_clim_slider.reset()
        sp_clim_slider.reset()

    button_reset.on_clicked(reset)

    def done(event):
        with open('parameters_text.txt', 'w') as file:
            file.write(str(x0_slider.val) + ' ' +
                       str(y0_slider.val) + ' ' +
                       str(deltax_slider.val) + ' ' +
                       str(deltay_slider.val) + ' ' +
                       str(theta_slider.val) + ' ')
        plt.close()

    button_done.on_clicked(done)

    plt.show()


def run(path_to_slits,
        hinode_B,
        p0=None,
        bounds=None,
        path_to_sunpy='/Users/jamescrowley/sunpy/',
        plot=True,
        save_coords=False,
        save_params=False,
        verbose=True,
        gui=True,
        remove_HMI=True):
    """
    Run:
        I want this to be the function which calls all the others to run the alignment, almost like a main

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
    :param gui:
        a bool, whether to use GUI in between initial and final fit to roughly vizualize alignment
    :param remove_HMI:


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
    closest_index, obstime = fetch_data(path_to_slits, path_to_sunpy, verbose)
    # TESTING july 7
    closest_index = [closest_index[sizex // 2]] * sizex
    all_HMI_data, hmix, hmiy, hmi_wcs = read_in_HMI()

    if verbose:
        print('Fido successfully downloaded HMI data.')

        print(50 * '-')
        print('Performing Initial Rough Alignment')

    initial_xcen = fits.open(path_to_slits + '/' + sorted(os.listdir(path_to_slits))[0])[0].header['XCEN']
    initial_ycen = fits.open(path_to_slits + '/' + sorted(os.listdir(path_to_slits))[0])[0].header['YCEN']

    # p0 = [initial_xcen, initial_ycen, 0, 0, 2.5]

    if bounds is None:
        bounds = [(-40, 40), (-40, 40), (-0.2, 0.2), (-0.2, 0.2), (-4, 4)]
    else:
        bounds = bounds

    if gui:
        xcen = (p0[0]) * u.arcsec
        ycen = (p0[1]) * u.arcsec
        deltax = p0[2]
        deltay = p0[3]
        theta = p0[4]

        initial_coords = get_coordinates(slits=slits,
                                         deltax=deltax,
                                         deltay=deltay,
                                         path_to_slits=path_to_slits,
                                         sizey=sizey,
                                         theta=theta)
        initialx = initial_coords[:, :, 0]
        initialy = initial_coords[:, :, 1]

        initialx = initialx + xcen.value
        initialy = initialy + ycen.value

        Nx = hinode_B.shape[0]
        Ny = hinode_B.shape[1]

        output = np.zeros((initialx.shape[0], initialx.shape[1], 2))

        output[:Nx, :Ny, 0] = initialx[:Nx, :Ny]
        output[:Nx, :Ny, 1] = initialy[:Nx, :Ny]

        output = np.sort(output, axis=0)

        show_gui(p0,
                 all_HMI_data[:, :, 0],
                 hmix.value,
                 hmiy.value,
                 output,
                 hinode_B)

        with open('parameters_text.txt', 'r') as file:
            content = file.read()
            numbers_str = content.split()  # Split by any whitespace
            p0 = [float(num) for num in numbers_str]

        if verbose:
            print(50 * '-')
            print('Performing Final Fit')

    # Code works better with an initial guess of parameters. If not using GUI, do a rough fit:
    else:
        temp_bounds = [(-40, 40), (-40, 40), (p0[2] - 0.05, p0[2] + 0.05), (p0[3] - 0.05, p0[3] + 0.05), (p0[4], p0[4])]
        converged, p0 = minimize(p0,
                                 slits_sorted,
                                 path_to_slits,
                                 all_HMI_data,
                                 hmi_wcs,
                                 hinode_B,
                                 closest_index,
                                 sizes,
                                 obstime,
                                 temp_bounds)

    if verbose:
        print('Initial Rough Alignment Complete.')
        print('Estimate of parameters: ' + str(p0))
        print(50 * '-')

    converged, parameters = minimize(p0,
                                     slits_sorted,
                                     path_to_slits,
                                     all_HMI_data,
                                     hmi_wcs,
                                     hinode_B,
                                     closest_index,
                                     sizes,
                                     obstime,
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
                                                          hmi_wcs,
                                                          hinode_B,
                                                          closest_index,
                                                          sizes,
                                                          obstime,
                                                          False)

        plot_and_viz_compare(hinode_B, final_HMI)

    all_HMI_files = os.listdir(path_to_sunpy + '/data/HMI/align/')

    # remove the HMI maps
    if remove_HMI:
        for file in all_HMI_files:
            os.remove(path_to_sunpy + '/data/HMI/align/' + file)

    # save the coordinate, and any other requested information
    if save_coords:  # if saving, create the final coordinate arrays.

        deltax = parameters[2]
        deltay = parameters[3]

        theta = parameters[4]

        final_coordinates = get_coordinates(slits=slits,
                                            deltax=deltax,
                                            deltay=deltay,
                                            path_to_slits=path_to_slits,
                                            sizey=sizey,
                                            theta=theta)

        finalx = final_coordinates[:, :, 0] + parameters[0]
        finaly = final_coordinates[:, :, 1] + parameters[1]

        Nx = hinode_B.shape[0]
        Ny = hinode_B.shape[1]

        output = np.zeros((finalx.shape[0], finalx.shape[1], 2))

        output[:Nx, :Ny, 0] = finalx[:Nx, :Ny]
        output[:Nx, :Ny, 1] = finaly[:Nx, :Ny]

        output = np.sort(output, axis=0)

        fits.writeto('coordinates.fits', output, overwrite=True)
    if save_params:
        fits.writeto('parameters.fits', parameters, overwrite=True)
