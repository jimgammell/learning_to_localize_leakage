import argparse
from pathlib import Path

from init_things import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=True, type=Path)
    parser.add_argument('--dest', type=Path, default=None)
    parser.add_argument('--do-attack-evaluation', default=False, action='store_true')
    append_directory_clargs(parser)
    args = parser.parse_args()

if __name__ == '__main__':
    main()