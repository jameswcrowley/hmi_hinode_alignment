import data_utils as du
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',
                        dest='plot',
                        type=bool,
                        required=False,
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
    parser.add_argument('--output_format',
                        dest='output_format',
                        type=list,
                        required=True,
                        help='output format to save the coords. if only vizualizing, use []. otherwise, accepted'
                             'arguments are "HPCx", "HPCy", "hinodeB"')

    arg = parser.parse_args()

    plot = arg.plot  # TODO: plot option not working... figure out why and fix this.
    path_to_slits = arg.path_to_slits
    name_hinode_B = arg.name_hinode_B
    path_to_sunpy = arg.path_to_sunpy
    output_format = arg.output_format

    hinode_model = du.fits.open(name_hinode_B)[0].data
    hinode_B = hinode_model

    # --- interpolating ---
    du.run(path_to_slits,
           hinode_B,
           bounds=None,
           path_to_sunpy=path_to_sunpy,
           output_format=output_format,
           plot=plot)


if __name__ == '__main__':
    main()
