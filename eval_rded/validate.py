from argument import args
from validation.main import main as valid_main

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    valid_main(args)