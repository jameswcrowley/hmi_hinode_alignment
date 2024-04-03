import data_utils as du
import argparse


def main():

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',
                        dest='plot',
                        type=str2bool,
                        help='whether or not to plot and visually compare result.')
    parser.add_argument('--path_to_slits',
                        dest='path_to_slits',
                        type=str,
                        required=True,
                        help='path to raw, unpacked Hinode slits')
    parser.add_argument('--name_hinode_B',
                        dest='name_hinode_B',
                        type=str,
                        required=True,
                        help='path/name of hinode B field to be aligned')
    parser.add_argument('--path_to_sunpy',
                        dest='path_to_sunpy',
                        type=str,
                        required=True,
                        help='path to sunpy. On my computer it is /Users/jamescrowley/sunpy/')
    parser.add_argument('--save_coords',
                        dest='save_coords',
                        type=str2bool,
                        help='Whether or not to save the final coordinates, in HPC x/y.')

    parser.add_argument('--save_params',
                        dest='save_params',
                        type=str2bool,
                        help='Whether or not to save the final fitted parameters, in the order '
                             '[x_cen, y_cen, p_x, p_y, theta].')
    parser.add_argument('--verbose',
                        dest='verbose',
                        type=str2bool,
                        help='If you want code to print updates. Default = True')


    arg = parser.parse_args()

    plot = arg.plot
    path_to_slits = arg.path_to_slits
    name_hinode_B = arg.name_hinode_B
    path_to_sunpy = arg.path_to_sunpy
    save_coords = arg.save_coords
    save_params = arg.save_params
    verbose = arg.verbose

    if verbose is None:
        verbose = True

    hinode_model = du.fits.open(name_hinode_B)[0].data
    hinode_B = hinode_model

    # --- interpolating ---
    du.run(path_to_slits,
           hinode_B,
           bounds=None,
           path_to_sunpy=path_to_sunpy,
           plot=plot,
           save_coords=save_coords,
           save_params=save_params,
           verbose=verbose)

if __name__ == '__main__':
    main()
