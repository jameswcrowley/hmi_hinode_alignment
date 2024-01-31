import data_utils as du
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',
                        dest='plot',
                        type=bool,
                        required=True,
                        help='whether or not to plot and visually compare result.')

    arg = parser.parse_args()

    plot = arg.plot

    # --- interpolating ---

    if not plot:
        #plot
    else:
        # save/return

    if __name__ == '__main__':
        main()