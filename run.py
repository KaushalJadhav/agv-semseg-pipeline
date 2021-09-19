from utils.trainer import evaluate
import train, test
from utils.config import process_config
from utils.preprocessing import encode_data
from dataloader.cityscapes import CityScapes, load_data, get_data_array, get_data
from dataloader.cityscapesCutmix import CityScapesCutmix, load_data, get_data_array, get_data
from train import train, validate, finalize

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

    train_X, train_y, val_X, val_y, test_X = get_data_array(config["train_X_path"], config["train_y_path"], config["val_X_path"], config["val_y_path"], config["test_X_path"])
    train_y_encoded, val_y_encoded = encode_data(train_y, val_y)
    train_data, val_data = get_data(train_X, train_y_encoded, val_X, val_y_encoded)
    train_loader, val_loader = load_data(train_data, val_data)

    train()
    validate()
    finalize()



    



if __name__ == '__main__':
    main()