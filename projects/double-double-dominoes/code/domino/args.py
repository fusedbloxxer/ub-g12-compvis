import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='double-double-dominoes', allow_abbrev=True)

    parser.add_argument('path')

    args = parser.parse_args()

    return args
