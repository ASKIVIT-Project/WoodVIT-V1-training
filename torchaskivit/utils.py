import os
import sys
import time
import random
import io
import contextlib
import glob
import argparse
# import spectral

import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
from torchvision.transforms import v2 #v2 is libary of torchvision transformations
from torchsummary import summary
from torch.optim import lr_scheduler

from . import cnn_classes as cnn
from . import custom_transforms as ct


def get_time_elapsed(start_time):
    """
    Calculates the time elapsed since the given start time and returns it in 
    the format "hh:mm:ss".

    Parameters
    ----------
    start_time : float
        The start time in seconds.

    Returns
    -------
    str : The time elapsed in the format "hh:mm:ss".
    """
    time_elapsed = time.time() - start_time
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed - (hours * 3600)) // 60)
    seconds = float(time_elapsed % 60)

    return f"{hours:02d}h {minutes:02d}m {seconds:05.2f}s"


def set_seed(seed_value):
    """
    Sets seed if reproducibility is favored.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    # Set to False, otherwise reproducibility is not guaranteed.
    torch.backends.cudnn.benchmark = False


def lp(message, P):
    """
    Wrapper to be able to write to log file and print to console at the same
    time.
    """
    with open(P["LOG_PATH"], "a", encoding="utf-8") as f:
        f.write(message + "\n")

    print(message)


def print_log(P, d, args):
    """
    Print log.

    Parameters
    ----------
    P : dict
        Constant parameters.
    d : dict
        Later definded parameters.
    """

    # Prepare model print for log by removing the first and last line, and
    # removing the first two characters of each line.
    model_summary = str(d["model"])
    lines = model_summary.split("\n")[1:-1]
    modified_lines = [line[2:] for line in lines]
    model_info = "\n".join(modified_lines)

    # Capture the torchsummary output, otherwise it will be printed to the
    # console immediately, and not at the location it should be in the log.
    summary_buffer = io.StringIO()
    with contextlib.redirect_stdout(summary_buffer):
        summary(d['model'], (len(P['SELECTED_CH']),
                P["INPUT_SIZE"], P["INPUT_SIZE"]))
    summary_output = summary_buffer.getvalue()

    # Modifying transformation output for better readability.
    transforms = str(P["TRANSFORM_TRAIN"])
    transforms_lines = transforms.split("\n")[1:-1]
    transforms_modified = [line[6:] for line in transforms_lines]
    transforms_info = "\n".join(transforms_modified)

    # Log message.
    output_message = f"""{P["START_TIME"].strftime("%d.%m.%Y %H:%M:%S")}

Versions
--------
Python: {sys.version.split()[0]}
PyTorch: {torch.__version__}
CUDA: {torch.version.cuda}
cuDNN: {torch.backends.cudnn.version()}

CUDA is available: {torch.cuda.is_available()}
Using device: {P['DEVICE']}


Model
-----
{d['model'].__class__.__name__}

{model_info}


{summary_output}
Note: Paramter count matches models from SPIE paper for 428 input channels.


Parameters
----------
channels: [{P["SELECTED_CH"][0]}, ..., {P["SELECTED_CH"][-1]}]
num of channels: {len(P["SELECTED_CH"])}
lr: {args.lr}
batch size: {P["BATCH_SIZE"]}
num epochs: {P["NUM_EPOCHS"]}
early stopping: after {P["PATIENCE"]}
min delta: {P["MIN_DELTA"]}
pin memory: {P["PIN_MEMORY"]}
num workers: {P["NUM_WORKERS"]}
seed: {P["SEED"]}
criterion: {d["criterion"].__class__.__name__}
optimizer: {d["optimizer"].__class__.__name__}
scheduler: {d["scheduler"].__class__.__name__}


Transformations
---------------
{transforms_info}
    """

    return output_message


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def save_model(
        P,
        d,
        args,
        train_acc,
        val_acc,
        val_loss,
        epoch,
        delete=False,
        add="None"
):
    """
    Save model and additional information to disk.

    Parameters
    ----------
    P : dict
        Constant parameters.
    d : dict
        Later definded parameters.
    train_acc : float
        The training accuracy.
    val_acc : float
        The validation accuracy.
    val_loss : float
        The validation loss.
    epoch : int
        The current epoch.
    delete : bool, optional
        If True, previously saved model with same start time and model name
        will be deleted. The default is False.
    """

    model = d["model"].__class__.__name__

    metadata = {
        "model": model,
        "sensor": P["SENSOR"],
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_acc": train_acc,
        "epoch": epoch,
        "start_time": P["START_TIME"],
        "hyperparameters": {
            "lr_max_or_start": args.lr,
            "batch_size": P["BATCH_SIZE"],
            "num_epochs": P["NUM_EPOCHS"],
            "seed": P["SEED"],
            "criterion": d["criterion"].__class__.__name__,
            "optimizer": d["optimizer"].__class__.__name__,
        }
    }

    checkpoint = {
        "state_dict": d["model"].state_dict(),
        # Saving optimizer state: Not possible, otherwise file size is too big
        # for GitLab. Not needed anyway right now.
        # "optimizer_dict": d["optimizer"].state_dict(),
        "metadata": metadata
    }

    txt = (f"{P['START_TIME_STR']}_{model}_{P['SENSOR']}_lr_{args.lr}_"
           f"vl_{val_loss:.4f}_va_{val_acc:.1f}"
           f"_ta_{train_acc:.1f}_e_{epoch+1}_{add}.pth")
    path = os.path.join(P["DIR_SAVEMODELS"], txt)

    if delete:
        pattern = os.path.join(
            P["DIR_SAVEMODELS"], f"{P['START_TIME_STR']}_{model}*.pth")
        for filename in glob.glob(pattern):
            os.remove(filename)

    torch.save(checkpoint, path)
    lp(f"Model saved to {txt}", P)


def update_model_with_test_results(P, d, test_acc, test_loss):
    """
    Update the saved model checkpoint with test accuracy and test loss by matching the start time string.

    Parameters
    ----------
    test_acc : float
        The test accuracy to be added.
    test_loss : float
        The test loss to be added.
    """
    # Construct the pattern to find the correct file based on the start time
    pattern = os.path.join(P["DIR_SAVEMODELS"], f"{P['START_TIME_STR']}_*.pth")
    files = glob.glob(pattern)

    if not files:
        print("No matching files found.")
        return

    # Load, update, and save each matching file
    for file_path in files:
        # Load the checkpoint
        checkpoint = torch.load(file_path)

        # Update the metadata with test_acc and test_loss
        checkpoint["metadata"]["test_acc"] = test_acc
        checkpoint["metadata"]["test_loss"] = test_loss

        # Save the checkpoint back to the same file
        torch.save(checkpoint, file_path)
        print(f"Model updated with test results and saved to {file_path}")


def parse_args(P):
    """
    When you add arguments here, make sure you also add them to the log message
    by replacing the P["..."] with args.example
    """
    parser = argparse.ArgumentParser(description="Torch Training Script")

    parser.add_argument("--model", type=str, default=P["MODEL"])
    parser.add_argument("--sensor", type=str, default=P["SENSOR"])
    parser.add_argument("--lr", type=float, default=P["LEARNING_RATE"])

    args = parser.parse_args()

    return args


def choose_model(P, model_name):
    # Default to CNN1 if the specified model is not found.
    cnn_name = "CNN1" if not hasattr(cnn, model_name) else model_name
    cnn_class = getattr(cnn, cnn_name)

    # Create an instance of the specified model.
    model = cnn_class(P["SELECTED_CH"], P["INPUT_SIZE"],
                      num_classes=P["NUM_CLASSES"])

    return model


def transform_pipeline(
        P,
        model_name,
        is_train=True,
        resize=None,
        BGR2RGB=False,
        scale=False,
        rgb=False,
        overide_stats=None,
) -> List:
    """
    Creates a transformation pipeline for preprocessing data, tailored to the
    specified model and training status.

    Parameters
    ----------
    P : dict
        A dictionary containing constants and parameters, including statistics
        for normalization.
    model_name : str
        The name of the model for which the pipeline is being created. This
        determines the normalization statistics to use.
    is_train : bool, optional
        Flag indicating whether the pipeline is for training data. This affects
        the inclusion of data augmentation transforms. The default is True.

    Returns
    -------
    list: A list of transformations to be applied to the data.
          Has to be wrapped in v2.Compose to be used as a transformation.
    """
    # Define dictionary to map model names to corresponding statistics keys.
    model_stats = {
        "ResNet50": "ImageNet",
        "SqueezeNet": "ImageNet",
        "default": "ASKIVIT_V1.5",
        "EfficientNetB3": "ImageNet",
        "EfficientNetB3_Complex": "ImageNet",
        "EfficientNetB3_Simple": "ImageNet",
    }

    # Use the model name to get the appropriate statistics key, defaulting to
    # "ASKIVIT_V1.5" if the model name is not found.
    stats_key = model_stats.get(model_name, model_stats["default"])

    # Retrieve means and stds using the determined stats key
    means = P["STATISTICS"][stats_key]["MEANS"]
    stds = P["STATISTICS"][stats_key]["STDS"]

    if overide_stats is not None:
        means = overide_stats["MEANS"]
        stds = overide_stats["STDS"]

    """ Create the transformation pipeline. """
    transforms = []

    transforms.extend([
        ct.SelectChannels(list(P["SELECTED_CH"])),
    ])

    if BGR2RGB and P["SENSOR"] == "RGB":
        transforms.extend([
            ct.BGR2RGB(),
        ])
            
    # Base transforms needed to handle dataset.
    transforms.extend([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=scale),
    ])

    # Add training specific transforms to improve generalization.
    if is_train:
        if rgb:
            transforms.extend([
                v2.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.05, 0.05)),
            ])
        transforms.extend([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            # v2.RandomRotation(degrees=(0, 180)),
        ])

    if resize is not None:
        transforms.extend([
            v2.Resize((resize, resize), antialias=True),
        ])

    # Add normalization transform, specific to ASKIVIT, or the dataset the
    # transfer model was trained on.
    transforms.extend([
        ct.NormalizeWrapped(means, stds),
    ])

    return transforms


def get_ch_for_sensor(sensor: str) -> range:
    """
    ---> FOR ASKIVIT V2 <---
    Returns the range of channel indices for a given sensor type.

    Parameters
    ----------
    sensor : str
        The type of sensor. Should be one of "RGB", "NIR", "IR", "THz", or "EF"

    Returns
    -------
    range
        The range of channel indices corresponding to the sensor type.

    Raises
    ------
    ValueError
        If an invalid sensor type is provided.
    """

    sensor_ranges = {
        "RGB": range(0, 3),
        "NIR": range(3, 227),
        "IR": range(227, 517),
        "THz": range(517, 717),
        "EF": range(0, 717),
    }

    if sensor in sensor_ranges:
        return sensor_ranges[sensor]
    else:
        raise ValueError(
            "Invalid sensor type. Must be one of: RGB, NIR, IR, THz")
    

def get_ch_for_sensor_v2(sensor: str) -> range:
    """
    ---> FOR ASKIVIT V2 <---
    Returns the range of channel indices for a given sensor type.

    Parameters
    ----------
    sensor : str
        The type of sensor. Should be one of "RGB", "NIR", "Thermo", "THz", or "EF"

    Returns
    -------
    range
        The range of channel indices corresponding to the sensor type.

    Raises
    ------
    ValueError
        If an invalid sensor type is provided.
    """

    sensor_ranges = {
        "RGB": range(0, 3),
        "NIR": range(0, 224),
        "Thermo": range(0, 1),
        "THz": range(0, 200),
        "EF": range(0, 428),
    }

    if sensor in sensor_ranges:
        return sensor_ranges[sensor]
    else:
        raise ValueError(
            "Invalid sensor type. Must be one of: RGB, NIR, Thermo, THz")


def get_res_for_sensor_v2(sensor: str):
    """
    ---> FOR ASKIVIT V2 <---
    Returns the data resolution for a given sensor type.

    Parameters
    ----------
    sensor : str
        The type of sensor. Should be one of "RGB", "NIR", "Thermo", "THz", or "EF"
    """

    sensor_resolutions = {
            "RGB": 321,
            "NIR": 53,
            "Thermo": 53,
            "THz": 50,
            "EF": 50
        }
    
    if sensor in sensor_resolutions:
        return sensor_resolutions[sensor]
    else:
        raise ValueError("Invalid sensor type. Must be one of: RGB, NIR, Thermo, THz")


def warmup_scheduler(optimizer, initial_lr, warmup_epochs=5, warmup_start_lr=1e-6, decay_factor=0.95):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warm-up
            lr = warmup_start_lr + \
                (initial_lr - warmup_start_lr) / warmup_epochs * epoch
        else:
            # Apply decay after warm-up
            lr = initial_lr * (decay_factor ** (epoch - warmup_epochs))
        return lr / initial_lr

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


# def img_loader_nir(filepath: str, interleave='bip', writable=True) -> np.ndarray:
#     try:
#         img = spectral.io.envi.open(f'{filepath}.hdr', f'{filepath}.raw')
#         img = img.open_memmap(interleave=interleave, writable=writable)

#     except:
#         raise Exception("Image couldn't be loaded")

#     return img


def get_actual_label_ASKIVIT_V2(filepath):
    """ 
    DEPRECATED: Modified new version of this function can be found in 

    ---> evalutils.py <---

    Returns not only the main label, but also the sublabel.

    """
    filename = os.path.basename(filepath)
    label = "_".join(filename.split("_")[-3:-1])

    # Map label to two main classes.
    main_label = label.split("_")[0]
    if main_label == "Holz":
        return 0
    elif main_label == "Nicht Holz" or main_label == "Background":
        return 1


def get_actual_label_ASKIVIT_V1(filepath):
    """ 
    DEPRECATED: Modified new version of this function can be found in 

    ---> evalutils.py <---

    Returns not only the main label, but also the sublabel.

    """
    filename = os.path.basename(filepath)
    label = "_".join(filename.split(".")[0].split("_")[-2:])

    # Map label to two main classes.
    main_label = label.split("_")[0]
    if main_label == "Holz":
        return 0
    elif main_label == "Nicht-Holz":
        return 1
