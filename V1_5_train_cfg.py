import os
import sys
import json
import datetime

import torch


############################################################################################################################################
CWD = r"C:\Users"
DATA_DIR = r"D:\Askivit Daten\ASKIVIT_1_5"
SAVE_DIR = r"C:\Users"
MODEL_SAVE_DIR = SAVE_DIR

#path to means and stds data created in data analysis (analysis_ASKIVIT_V1_5.py)
FILE_STATS = r"C:\Users\ASKIVIT_1_5_means_and_stds_channel_range_0_to_717.json"
############################################################################################################################################
DATA_DIR_TEST = os.path.join(DATA_DIR, "patches_test")
DATA_DIR_TRAIN = os.path.join(DATA_DIR, "patches_train")
DATA_DIR_VAL = os.path.join(DATA_DIR, "patches_val")
############################################################################################################################################
TB_WRITER = os.path.join(SAVE_DIR, r"Tensorboard_Summarys")
os.makedirs(TB_WRITER, exist_ok=True)
############################################################################################################################################

sys.path.append(CWD)
import torchaskivit.utils as utils


""" SET PARAMETERS """
MODEL = "CNN1BN"
INPUT_SIZE = 50

NUM_CLASSES = 2

SENSOR = "EF"
SELECTED_CH = utils.get_ch_for_sensor(SENSOR)

LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
BATCH_SIZE = 32
PIN_MEMORY = True
NUM_WORKERS = 6
SEED = None # Set to None if seed is not wanted
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_TIME = datetime.datetime.now()
START_TIME_STR = START_TIME.strftime("%y%m%d_%H%M%S")
LOG_PATH = os.path.join(SAVE_DIR, f"{START_TIME_STR}_training.log")
WRITER_PATH = os.path.join(TB_WRITER, f"{START_TIME_STR}_tb_summary")


""" EARLY STOPPING PARAMETERS """
PATIENCE = 15
MIN_DELTA = 0.005


""" CALCULATE ADDITIONAL PARAMETERS """
# Calculate weights to be used by the loss function, to achieve class balance.
# Count obtained from "ba_niemoeller/09_Code/01_dataset_analysis/other
# /check_askivit_1_3.ipynb". Counts are Holz = 4952, Nicht-Holz = 11060.
CLASS_COUNTS = [3903, 8720]
WEIGHTS = [sum(CLASS_COUNTS) / c for c in CLASS_COUNTS]
LOSS_WEIGHTS = torch.FloatTensor(WEIGHTS).cuda()


# Get mean and std of training set.
with open(FILE_STATS, "r") as f:
    stats = json.load(f)

MEANS = stats["means"]
STDS = stats["stds"]

# Slice means and stds to only include selected channels.
SLICED_MEANS = torch.tensor([MEANS[i] for i in SELECTED_CH])
SLICED_STDS = torch.tensor([STDS[i] for i in SELECTED_CH])

# Dict with means and stds for different models.
STATISTICS = {
    #"ASKIVIT_V1.3": {
    #    "MEANS": SLICED_MEANS,
    #    "STDS": SLICED_STDS,
    #},
    "ImageNet": {
        "MEANS": [0.485, 0.456, 0.406],
        "STDS": [0.229, 0.224, 0.225],

    },
    "ASKIVIT_V1.5": {
        "MEANS": SLICED_MEANS,
        "STDS": SLICED_STDS,
    }
}


""" LABELS TO INDICES """
LABEL_IDX = {
    "Nicht-Holz_Hintergrund": 0,
    "Holz_Massivholz": 1,
    "Holz_Sperrholz": 2,
    "Holz_Spanplatte": 3,
    "Holz_Mitteldichte-Faserplatte": 4,
    "Nicht-Holz_Metall": 5,
    "Nicht-Holz_Kunststoff": 6,
    "Nicht-Holz_Mineralik": 7,
    "Nicht-Holz_Polster": 8,
    "Nicht-Holz_Holz-verdeckt-durch-Polster": 9,
    "Nicht-Holz_Holz-verdeckt-durch-Karton": 10,
    "Nicht-Holz_Holz-verdeckt-durch-Kunststoff": 11,
    "Nicht-Holz_Holz-verdeckt-durch-Mineralik": 12,
    "Nicht-Holz_Metall-verdeckt-durch-Holz": 13,
    "Nicht-Holz_Metall-verdeckt-durch-Polster": 14,
    "Nicht-Holz_Metall-verdeckt-durch-Kunststoff": 15,
}


""" BUNDLE PARAMETERS"""
PARAMS = {
    "CWD": CWD,
    "DIR_TRAINSET": DATA_DIR_TRAIN,
    "DIR_VALSET": DATA_DIR_VAL,
    "DIR_TESTSET": DATA_DIR_TEST,
    "DIR_SAVEMODELS": MODEL_SAVE_DIR,
    "TB_WRITER": TB_WRITER,
    "FILE_STATS": FILE_STATS,

    "MODEL": MODEL,
    "INPUT_SIZE": INPUT_SIZE,

    "SENSOR": SENSOR, # "RGB", "NIR", "IR", "THz"
    "NUM_CLASSES": NUM_CLASSES,
    "SELECTED_CH": SELECTED_CH,
    "LEARNING_RATE": LEARNING_RATE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "PIN_MEMORY": PIN_MEMORY,
    "NUM_WORKERS": NUM_WORKERS,
    "SEED": SEED,
    "DEVICE": DEVICE,

    "START_TIME": START_TIME,
    "START_TIME_STR": START_TIME_STR,
    "LOG_PATH": LOG_PATH,
    "WRITER_PATH": WRITER_PATH,

    "PATIENCE": PATIENCE,
    "MIN_DELTA": MIN_DELTA,

    "LOSS_WEIGHTS": LOSS_WEIGHTS,

    "MEANS": MEANS,
    "STDS": STDS,
    "SLICED_MEANS": SLICED_MEANS,
    "SLICED_STDS": SLICED_STDS,

    "STATISTICS": STATISTICS,

    "LABEL_IDX": LABEL_IDX,
}