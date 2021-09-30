
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


    # Not sure if config object could be logged
    wandb.init(
        name=config.exp_name,
        config=config,
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
                       train_iou,
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

def load_model_wandb(model_path,run_path:None):     # run_path is to be added in the config file
    """
    restore a model into the local run directory
    Args:
         filename (str): Path of saved model
         run_path (str): Referring to the earlier run from which to pull the file, 
                        formatted as '$ENTITY_NAME/$PROJECT_NAME/$RUN_ID'  or 
                        '$PROJECT_NAME/$RUN_ID' 
                        (default: current entity, project name, and run id)
                        Default- None
                        NOTE: resuming must be configured if run_path is not provided
    Returns:
         filename of the local copy

    """
    model=wandb.restore(model_path, run_path=run_path)
    # use the "name" attribute of the returned object
    # if your framework expects a filename, e.g. as in Keras
    return model.name

# Sving and loading using wandb.artifacts

def save_wandb_artifact(wandb_run,model_name,checkpoint_path,is_best=False):
# model_name can be taken from config. It is the name of the artifact 
# both the paths can be saved
    model_artifact = wandb.Artifact(
        model_name, 
        type="model",  
 # type is used to differentiate kinds of artifacts, used for organizational purposes.
 # For eg. "dataset", "model","result"
        description="trained Enet model",
 # The description is displayed next to the artifact version in the UI
 # Did not understand the role of metadata
        metadata=dict(wandb.config))
    if is_best:
        model_artifact.add_dir('best_path')
    else:
        model_artifact.add_dir('last_checkpoint_path')
    if is_best:
        model_artifact.add_file(checkpoint_path,name='best_path/'+checkpoint_path)
# NOTE- THE METRIC NAME HAS TO BE UPDATED
        wandb_run.log_artifact(artifact, aliases=['best'+metric_name])
    else:
        model_artifact.add_file(checkpoint_path,name='last_checkpoint_path/'+checkpoint_path)
        wandb_run.log_artifact(artifact, aliases=['latest'])

def load_wandb_artifact(wandb_run,model_name,is_best=False,root_download_path=None):
    if is_best:
# NOTE- THE METRIC NAME HAS TO BE UPDATED        
        artifact = wandb_run.use_artifact(model_name+':'+'best'+metric_name)
        datadir = artifact.download(root=root_download_path)
        #datadir is a path to a directory containing the artifact’s contents. 
        return datadir # NEEDS TO BE CHECKED
    else:
        # There is some error in the following line:
        # artifact = wandb_run.use_artifact(model_name+':latest)
        datadir = artifact.download(root=root_download_path)
        #datadir is a path to a directory containing the artifact’s contents. 
        return datadir # NEEDS TO BE CHECKED

    