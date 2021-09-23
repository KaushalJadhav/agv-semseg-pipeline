# from utils.trainer import evaluate
# import train, test
# from utils.config import process_config
# from utils.preprocessing import encode_data
# from dataloader.cityscapes import CityScapes, load_data, get_data_array, get_data
# from dataloader.cityscapesCutmix import CityScapesCutmix, load_data, get_data_array, get_data
# from train import train, validate, finalize
# Do not call any other method or class in run.py ~mradul2

from train import Train 
from test import ENetTestAgent

import argparse
from utils.config import process_config

def main():
    arg_parser = argparse.ArgumentParser(description="")

    arg_parser.add_argument(
        '--config',
        metavar='config_json_file',
        default='config/vanilla_enet.json',
        help='The Configuration file in json format')
    arg_parser.add_argument(
        '--mode',
        metavar='mode_of_running',
        default='train',
        help='Mode of running: train or test')
    arg_parser.add_argument(
        '--wandb_id',
        metavar='api_key_wandb',
        default=None,
        help='Enter your wandb API key')
    arg_parser.add_argument(
        '--cutmix',
        metavar='bool_for_cutmix',
        default=False,
        help='Enter the choice for training with cutmix')
    arg_parser.add_argument(
        '--epochs',
        metavar='num_of_epochs',
        default=25,
        help='Enter the number of epochs for training')
    arg_parser.add_argument(
        '--ignore_index',
        metavar='ignore_index',
        default=-100,
        help='Enter the index to ignore in training')
    arg_parser.add_argument(
        '--num_classes',
        metavar='num_of_classes',
        default=19,
        help='Enter the number of classes')
    arg_parser.add_argument(
        '--lr',
        metavar='learning_rate',
        default=0.0005,
        help='Enter the learning rate')
    arg_parser.add_argument(
        '--gamma',
        metavar='gamma',
        default=0.98,
        help='Enter the value of gamma')
    arg_parser.add_argument(
        '--momentum',
        metavar='momentum',
        default=0.98,
        help='Enter the value of momentum')
    arg_parser.add_argument(
        '--weight_decay',
        metavar='weight_decay',
        default=0.0002,
        help='Enter the value of weight_decay')
    arg_parser.add_argument(
        '--train_bs',
        metavar='train_batch_size',
        default=16,
        help='Enter the value of train_batch_size')
    arg_parser.add_argument(
        '--check_path',
        metavar='checkpoint_path',
        default="",
        help='Enter the path for saving checkpoint model')
    arg_parser.add_argument(
        '--best_path',
        metavar='best_model_path',
        default="",
        help='Enter the path for saving bets model')


    # Similary more arguments can be parsed as per the need 
    args = arg_parser.parse_args()

    # Parse the config json file
    config = process_config(args.config)

    # Set configs provided
    config.mode = args.mode
    config.wandb_id = args.wandb_id
    config.cutmix = args.cutmix
    config.max_epoch = args.epochs
    config.ignore_index = args.ignore_index
    config.num_classes = args.num_classes
    config.learning_rate = args.lr
    config.gamma = args.gamma
    config.momentum = args.momentum
    config.weight_decay = args.weight_decay
    config.train_batch_size = args.train_bs
    config.checkpoint_path = args.check_path
    config.best_model_path = args.best_path


    # Not like this : 
    # All these stuffs are initialized in train.py class only
    # train_X, train_y, val_X, val_y, test_X = get_data_array(config["train_X_path"], config["train_y_path"], config["val_X_path"], config["val_y_path"], config["test_X_path"])
    # train_y_encoded, val_y_encoded = encode_data(train_y, val_y)
    # train_data, val_data = get_data(train_X, train_y_encoded, val_X, val_y_encoded)
    # train_loader, val_loader = load_data(train_data, val_data)
    # train()
    # validate()
    # finalize() ~mradul2

    # Check the mode
    if config.mode=="train":
        # Create a train class instance and call thier methods like this 
        agent = Train(config)
        agent.forward()
    if config.mode=="test":
        agent = Test(config)
        agent.forward()
    else: 
        print("Invalid mode provided!")


    



if __name__ == '__main__':
    main()