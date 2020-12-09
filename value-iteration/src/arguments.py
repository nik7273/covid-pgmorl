import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='RL')

    # MORL parameters
    parser.add_argument('--obj-num',
        type=int,
        default=2,
        help='number of objectives')
    parser.add_argument('--num-env-steps',
        type=int,
        default=5e6,
        help='number of environment steps to train (default: 5e6)')
    parser.add_argument('--num-steps',
        type=int,
        default=1,
        help='number of steps (default: 1)')
    parser.add_argument('--num-tasks',
        type=int,
        default=11,
        help='number of rl task in each epoch')
    parser.add_argument('--total-num-updates',
        type=int,
        default=20,
        help='number of iterations of MORL (Alg 1)'
    )
    parser.add_argument('--seed', 
        type=int, 
        default=0, 
        help='random seed (default: 1)')
    parser.add_argument('--min-weight',
        type=float,
        default=0.0,
        help='minimum of weight range')
    parser.add_argument('--max-weight',
        type=float,
        default=1.0,
        help='maximum of weight range')
    parser.add_argument('--delta-weight',
        type=float,
        default=0.05,
        help='granularity of weight combinations in warm-up stage')
    parser.add_argument('--warmup-iter',
        type=int,
        default=3,
        help='number of RL iterations to run for warm up')
    parser.add_argument('--update-iter',
        type=int, 
        default=3,
        help='number of RL iterations between evolutionary processes')
    parser.add_argument('--mopg-steps',
        type=int,
        default=10,
        help='number of steps to run mopg for (to get a policy)'
    )
    parser.add_argument('--eval-num',
        type=int,
        default=1,
        help='number of fitness evaluation times')
    parser.add_argument('--pbuffer-num',
        type=int,
        default=100,
        help='number of performance buffers')
    parser.add_argument('--pbuffer-size',
        type=int,
        default=2,
        help='size of each performance buffer')
    parser.add_argument('--num-weight-candidates',
        type=int,
        default=7,
        help='number of weight candiates for each population sample')
    parser.add_argument('--sparsity',
        default=1.0,
        type=float,
        help='alpha of sparsity metrics')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--dataset',
        default=None,
        help='.pt file with lockdown data'
    )

    return parser
