from train import Train 

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


    # Similary more arguments can be parsed as per the need 
    args = arg_parser.parse_args()

    # Parse the config json file
    config = process_config(args.config)

    # Set configs provided
    config.mode = args.mode
    config.wandb_id = args.wandb_id

    # Check the mode
    if config.mode=="train":
        # Create a train class instance and call thier methods like this 
        agent = Train(config)
        agent.forward()
        agent.summarize()
    if config.mode=="test":
        agent = Test(config)
        agent.forward()
    else: 
        print("Invalid mode provided!")


    



if __name__ == '__main__':
    main()