import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)

from arguments import get_parser
import morl
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# if there's overlap between args_list and commandline input, use commandline input
def solve_argv_conflict(args_list):
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith('-'):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i+1:]:
                        if not more_args.startswith('-'):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)

def main():
    print("Running")
    torch.set_default_dtype(torch.float64)

    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    
    if args.dataset is None:
        print("No Dataset given, make argument for dataset path")
        return -1
    
    # build saving folder
    save_dir = args.save_dir
    try:
        os.makedirs(save_dir, exist_ok = True)
    except OSError:
        pass

    morl.run(args)

if __name__ == "__main__":
    main()