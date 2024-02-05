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

    arg = parser.parse_args()

    plot = arg.plot
    path_to_slits = arg.path_to_slits
    name_hinode_B = arg.name_hinode_B

    hinode_model = du.fits.open(name_hinode_B)[0].data
    hinode_B = hinode_model[4, 10] * du.np.cos(hinode_model[6, 10] * du.np.pi / 180)

    # --- interpolating ---
    du.run(path_to_slits,
           hinode_B,
           bounds=[(25, 35), (15, 25), (0.9, 1.1), (0.9, 1.1), (-2, 2)])


if __name__ == '__main__':
    main()
