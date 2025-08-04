import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Process input parameters from schema.")

    parser.add_argument('--filter_mass', type=float, required=True,
                        help='Relative molecular mass screening thresholds in pretreatment (default: 200)', default=200)

    parser.add_argument('--predict_fpr', type=str2bool, default=True,
                        help='Use model output to predict fingerprint, and use fpr to compute sim (default: true)')

    parser.add_argument('--filter_formula', type=str2bool, default=False,
                        help='Filter by formula (default: false)')

    parser.add_argument('--inten_thresh', type=float, required=True,
                        help='Intensity threshold (default: 1)', default=1)

    parser.add_argument('--rt', type=float, required=True,
                        help='Retention time (default: 30)', default=30)

    parser.add_argument('--ppm', type=float, required=True,
                        help='PPM precision of data, used for matching signals between samples (default: 20)', default=20)

    parser.add_argument('--msdelta', type=float, required=True,
                        help='Msdelta (default: 0.01)', default=0.01)

    parser.add_argument('--if_merge_samples_byenergy', type=str2bool, default=False,
                        help='If merge samples by energy (default: false)')

    parser.add_argument('--min_mz_num', type=float, required=True,
                        help='Minimum number of mz (default: 2)', default=2)

    parser.add_argument('--remove_precursor', type=str2bool, default=True,
                        help='Remove precursor (default: true)')

    parser.add_argument('--output_num', type=float, required=True,
                        help='Output number of the most similar molecules (default: 20)', default=20)


    return parser.parse_args()