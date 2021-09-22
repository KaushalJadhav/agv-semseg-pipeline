
import wandb
import os
import numpy as np


def init_wandb(model, config) -> None:
    """
    Initialize project on Weights & Biases
    Args:
        model (Torch Model): Model for Training
        args (TrainOptions,optional): TrainOptions class (refer options/train_options.py). Defaults to None.
    """

    print("Wandb api key provided...")
    wandb.login(key=config.wandb_id)

    wandb.init(
        name=config.experiment,
        project=config.wandb_project,
        id=config.wandb_id,
        dir="./",
    )

    wandb.watch(model, log="all")


def wandb_log(train_loss, val_loss, epoch):
    """
    Logs the accuracy and loss to wandb
    Args:
        train_loss (float): Training loss
        val_loss (float): Validation loss
        train_acc (float): Training Accuracy
        val_acc (float): Validation Accuracy
        epoch (int): Epoch Number
    """

    wandb.log({
        'Loss/Training': train_loss,
        'Loss/Validation': val_loss,
    }, step=epoch)

def wandb_save_summary(valid_accuracy,
                       valid_iou,
                       train_accuracy,
                       valid_accuracy,
                       valid_results,
                       valid_X,
                       valid_y):
   
   
    """[summary]
    Args:
    """
    
    # To-do

    wandb.finish()


def save_model_wandb(save_path):
    """ 
    Saves model to wandb
    Args:
        save_path (str): Path to save the wandb model
    """

    wandb.save(os.path.abspath(save_path))