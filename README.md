# AGV DL Segmentation Pipeline

## Structure

```bash
├── config  
│   ├── vanilla_enet.json
├── data
├── dataloader
│   ├── __init__.py
│   ├── cityscapes.py
├── models
│   ├── __init__.py
│   ├── enet.py
├── utils
│   ├── __init__.py
│   ├── augmentation.py
│   ├── config.py
│   ├── device.py
│   ├── losses.py
│   ├── metric.py
│   ├── preprocessing.py
│   ├── saving.py
│   ├── scripts.py
│   ├── tester.py
│   ├── trainer.py
│   ├── wandb_utils.py
├── __init__.py
├── README.md
├── run.py
├── test.py
├── train.py              
```


## Usage

```bash
python3 run.py [-h] [--config CONFIG_FILE] [--mode MODE]
                    [--wandb_id "WANDB_API_KEY"] [--cutmix CUTMIX]
                    [--epochs NUM_OF_EPOCHS] [--ignore_index INDEX]
                    [--num_classes NUM_OF_CLASSES] [--lr LEARNING_RATE]
                    [--gamma GAMMA] [--momentum MOMENTUM]
                    [--weight_decay WEIGHT_DECAY] [--train_bs TRAINING_BATCH_SIZE]
                    [--check_path CHECKPOINT_PATH]
                    [--best_path BEST_MODEL_PATH]
```
