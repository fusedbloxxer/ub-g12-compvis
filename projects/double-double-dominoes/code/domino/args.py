import argparse
import pathlib as pb


def get_args() -> argparse.Namespace:
    # Create a single & simple parser
    parser = argparse.ArgumentParser()

    # Input/Output
    parser.add_argument('-i', '--input', required=True, action='store', type=pb.Path, help='An absolute or relative path to the folder which holds regular_tasks directory under it.')
    parser.add_argument('-o', '--output', required=True, action='store', type=pb.Path, help='An absolute or relative path to the desired output folder.')

    # Optional program arguments
    parser.add_argument('--grid', required=False, default='hough', choices=['hough', 'window'], help='Algorithm used to split the board grid.')
    parser.add_argument('--debug', required=False, action='store_true', default=False, help='Show additional output to understand what went wrong.')
    parser.add_argument('--train', required=False, action='store_true', default=False, help='Indicate that the dataset can also leverage the truth folder that is next to the input folder.')
    parser.add_argument('--show_matrix', required=False, action='store_true', default=False, help='Show the predicted matrix at each step of the pipeline.')
    parser.add_argument('--show_image', required=False, action='store_true', default=False, help='Show the inner process in obtaining the grid (very verbose).')
    args: argparse.Namespace = parser.parse_args()

    # Extract the arguments
    cell_splitter: str   = args.grid
    input_path: pb.Path  = args.input

    # Validate the arguments
    if not input_path.exists():
        raise FileNotFoundError(str(input_path.absolute()))
    if not input_path.is_dir():
        raise NotADirectoryError(str(input_path.absolute()))
    if cell_splitter not in ['hough', 'window']:
        raise ValueError('Cell splitter must be hough or window, got: {}'.format(cell_splitter))
    return args
