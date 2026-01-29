import os
import glob
import tqdm
import json

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

############################################################################################################################################
DATA_DIR = r"D:\Askivit Daten\ASKIVIT_1_5"
SAVE_DIR = r"C:\Users"
save_filename = "stats.txt"
save_filepath = os.path.join(SAVE_DIR, save_filename)
############################################################################################################################################
DATA_DIR_TEST = os.path.join(DATA_DIR, "patches_test\*.npy")
npy_files_test = glob.glob(DATA_DIR_TEST)
DATA_DIR_TRAIN = os.path.join(DATA_DIR, "patches_train\*.npy")
npy_files_train = glob.glob(DATA_DIR_TRAIN)
DATA_DIR_VAL = os.path.join(DATA_DIR, "patches_val\*.npy")
npy_files_val = glob.glob(DATA_DIR_VAL)

DATA_DIR_LIST = [DATA_DIR_TEST, DATA_DIR_VAL, DATA_DIR_TRAIN]
############################################################################################################################################


# find data shape for number of pixels: 50 and number of channels: 717 of train, test or val files
first_entry = npy_files_train[0]
data = np.load(first_entry)
data_shape = data.shape
with open(save_filepath, "w") as f:
    f.write(f"Shape of data: {data_shape}\n")
    f.write(f"\n")


# lists of all labels from test, train or val file names
# count subclasses and mainclasses for train, validation and test set each
true_label_counter = Counter()
gt_label_counter = Counter()
json_files_list = []
file_order = []

for directory in DATA_DIR_LIST:

    extracted_true_labels = []
    extracted_gt_labels = []
    true_label_counter.clear()
    gt_label_counter.clear()

    for file_path in glob.glob(directory):

        # get label out of file name
        filename = os.path.basename(file_path)
        label = "_".join(filename.split(".")[0].split("_")[5:])

        if "Non" in label:
            true_label = "_".join(label.split("_")[0:2])
            if not true_label in extracted_true_labels:
                extracted_true_labels.append(true_label)

            gt_label = "_".join(label.split("_")[2:])
            if not gt_label in extracted_gt_labels:
                extracted_gt_labels.append(gt_label)

        elif "Wood" in label:
            true_label = "_".join(label.split("_")[0:1])
            if not true_label in extracted_true_labels:
                extracted_true_labels.append(true_label)

            gt_label = "_".join(label.split("_")[1:])
            if not gt_label in extracted_gt_labels:
                extracted_gt_labels.append(gt_label)
        
        true_label_counter[true_label] += 1
        gt_label_counter[gt_label] += 1

    # saving the labels in json file
    data = {
        "true_labels": extracted_true_labels,
        "gt_labels": extracted_gt_labels,
        "true_label_quantities": dict(true_label_counter),
        "gt_label_quantities": dict(gt_label_counter)   
    }

    # naming the file
    json_filename = directory.split(os.sep)[-2].split("_")[1]
    json_filename_final = f"{json_filename}_lables_and_quantities.json"
    filepath = os.path.join(SAVE_DIR, json_filename_final)

    # list of created file names for later plotting
    json_files_list.append(os.path.join(SAVE_DIR, json_filename_final))
    file_order.append(json_filename)   #train, test, val in whatever order they are processed
    
    with open(filepath,"w") as f:
        json.dump(data, f, indent=4)


# number of train, test and validation files
num_train = len(npy_files_train)
num_test = len(npy_files_test)
num_val = len(npy_files_val)

# part of whole data in percent
percent_train = num_train / (num_train + num_test + num_val)
percent_val = num_val / (num_train + num_test + num_val)
percent_test = num_test / (num_train + num_test + num_val)

# save data in stats.txt
with open(save_filepath, "a") as f:  
    f.write(f"Train: {num_train} files with {percent_train:.3f}% of whole data.\n")
    f.write(f"Validate: {num_val} files with {percent_val:.3f}% of whole data.\n")
    f.write(f"Test: {num_test} files with {percent_test:.3f}% of whole data.\n")
    f.write(f"\n")


channels_ranges = [
    (0, 3),         #RGB
    (3, 227),       #NIR
    (227, 517),     #IR
    (517, 717),     #THz
    (0, 717)        #EarlyFusion
]

#shortening list for testing
#npy_files_train = npy_files_train[:5]

# min and max for the frequency bands to calculate norm_np (RGB 3, NIR 224, IR 290, THz 200) 
# a smaller norm_np might work better due to being more focused on most data points and cutting off the few outliers
for start, end in channels_ranges:
    abs_min = []
    abs_max = []
    total_pixels = 0
    num_channels = end - start
    sum_channels = np.zeros(num_channels)
    sumsq_channels = np.zeros(num_channels)

    for file in tqdm.tqdm(npy_files_train):
        data = np.load(file)
        #get max and min for norm_np
        temp_min = data[:, :, start:end].min(axis=(0, 1))
        abs_min.append(temp_min)
        temp_max = data[:, :, start:end].max(axis=(0, 1))
        abs_max.append(temp_max)

        # get sum and squared sum for each channel
        sum_channels += data[:, :, start:end].sum(axis=(0, 1))
        sumsq_channels += np.square(data[:, :, start:end]).sum(axis=(0, 1))
        
        total_pixels += data_shape[0] * data_shape[1]

    # calculate mean and std values for each channel
    means_for_range = sum_channels / total_pixels
    stds_for_range = np.sqrt(sumsq_channels / total_pixels - np.square(means_for_range))

    means_and_stds = {
        "means": means_for_range.tolist(),
        "stds": stds_for_range.tolist()
        }

    filename = f"{DATA_DIR.split(os.sep)[-1]}_means_and_stds_channel_range_{start}_to_{end}.json"
    with open(os.path.join(SAVE_DIR, filename), "w") as f:
        json.dump(means_and_stds, f)

    with open(save_filepath, "a") as f:
        f.write("Order: RGB, NIR, IR, THz\n")
        f.write(f"\n")
        f.write(f"Channel range: {start}...{end} norm_np: {np.min(abs_min), np.max(abs_max)}\n")
        f.write(f"\n")


# loading json file names with sub/main class stats
data = {}
for split, filename in zip(file_order, json_files_list):
    with open(filename, "r") as file:
        data[split] = json.load(file)

# get absolut quantities of SUBclasses from files
gt_labels = data[file_order[0]]["gt_labels"]

train_quantities_abs = [data["train"]["gt_label_quantities"][label] for label in gt_labels]
test_quantities_abs = [data['test']['gt_label_quantities'][label] for label in gt_labels]
val_quantities_abs = [data['val']['gt_label_quantities'][label] for label in gt_labels]

# plotting the bar chart for subclasses
ind = np.arange(len(gt_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(ind - width, train_quantities_abs, width, label='Train', color='#0000a7')
ax.bar(ind, val_quantities_abs, width, label='Validation', color='#eecc16')
ax.bar(ind + width, test_quantities_abs, width, label='Test', color='#920310')

ax.set_xlabel('Subclasses', fontsize=16)
ax.set_ylabel('Quantity', fontsize=16)
ax.set_title('Absolut distribution of subclasses in train, test and validation sets', fontsize=16)
ax.set_xticks(ind)
ax.set_xticklabels(gt_labels, rotation=55, ha="right")
max_quantity = max(max(train_quantities_abs), max(test_quantities_abs), max(val_quantities_abs))
ax.set_yticks(np.arange(0, max_quantity + 1, step=500))
ax.minorticks_on()
ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)

ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "subclass_distribution_absolut.pdf"), format='pdf', dpi=600, bbox_inches='tight')


# get percentage quantities of SUBclasses from files
gt_labels = data[file_order[0]]["gt_labels"]

train_quantities_percent = np.divide(train_quantities_abs, num_train)
test_quantities_percent = np.divide(test_quantities_abs, num_test)
val_quantities_percent = np.divide(val_quantities_abs, num_val)

# plotting the bar chart for subclasses
ind = np.arange(len(gt_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(ind - width, train_quantities_percent, width, label='Train', color='#0000a7')
ax.bar(ind, val_quantities_percent, width, label='Validation', color='#eecc16')
ax.bar(ind + width, test_quantities_percent, width, label='Test', color='#920310')

ax.set_xlabel('Subclasses', fontsize=16)
ax.set_ylabel('Relative quantity in proportion to whole set in %', fontsize=16)
ax.set_title('Relative distribution of subclasses in train, test and validation sets', fontsize=16)
ax.set_xticks(ind)
ax.set_xticklabels(gt_labels, rotation=55, ha="right")
ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)

ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "subclass_distribution_relative.pdf"), format='pdf', dpi=600, bbox_inches='tight')


# get absolut quantities of MAINclasses from files
true_labels = data[file_order[0]]["true_labels"]

train_quantities_abs = [data["train"]["true_label_quantities"][label] for label in true_labels]
test_quantities_abs = [data['test']['true_label_quantities'][label] for label in true_labels]
val_quantities_abs = [data['val']['true_label_quantities'][label] for label in true_labels]

# plotting the bar chart for mainclasses
ind = np.arange(len(true_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(ind - width, train_quantities_abs, width, label='Train', color='#0000a7')
ax.bar(ind, val_quantities_abs, width, label='Validation', color='#eecc16')
ax.bar(ind + width, test_quantities_abs, width, label='Test', color='#920310')

ax.set_xlabel('Mainclasses', fontsize=16)
ax.set_ylabel('Quantity', fontsize=16)
ax.set_title('Absolut distribution of mainclasses in train, test and validation sets', fontsize=16)
ax.set_xticks(ind)
ax.set_xticklabels(true_labels, rotation=55, ha="right", fontsize=16)
ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)

ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mainclass_distribution_absolut.pdf"), format='pdf', dpi=600, bbox_inches='tight')


# get relative quantities of MAINclasses from files
true_labels = data[file_order[0]]["true_labels"]

train_quantities_percent = np.divide(train_quantities_abs, num_train)
test_quantities_percent = np.divide(test_quantities_abs, num_test)
val_quantities_percent = np.divide(val_quantities_abs, num_val)

# plotting the bar chart for mainclasses
ind = np.arange(len(true_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(ind - width, train_quantities_percent, width, label='Train', color='#0000a7')
ax.bar(ind, val_quantities_percent, width, label='Validation', color='#eecc16')
ax.bar(ind + width, test_quantities_percent, width, label='Test', color='#920310')

ax.set_xlabel('Mainclasses', fontsize=16)
ax.set_ylabel('Relative quantity in proportion to whole set in %', fontsize=16)
ax.set_title('Relative distribution of mainclasses in train, test and validation sets', fontsize=16)
ax.set_xticks(ind)
ax.set_xticklabels(true_labels, rotation=55, ha="right", fontsize=16)
ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)

ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mainclass_distribution_relative.pdf"), format='pdf', dpi=600, bbox_inches='tight')