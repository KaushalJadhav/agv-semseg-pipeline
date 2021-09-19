import train, test
from utils.config import process_config

def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')

    args = arg_parser.parse_args()

    # Parse the config json file
    config = process_config(args.config)

    



if __name__ == '__main__':
    main()