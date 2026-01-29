import os
import sys
import json
import tqdm
import glob

import numpy as np
import torch
from torchvision.transforms import v2


# Filepaths to change if run on different machine.
CWD = r""
DATA_DIR = r"D:\Askivit Daten\ASKIVIT_1_5"
DIR_MODELS = r"C:\Users\"

## ****************************************************************************
without_verdeckt = False
## ****************************************************************************
# Choose model.
choose_model = "CNN1BN"
stats = "ASKIVIT_V1.5" # or "ASKIVIT_V1.4" "ASKIVIT_V2.0" or "ImageNet"
all_models = glob.glob(os.path.join(DIR_MODELS, "*.pth"))

# Create list of paths to test patches.
pattern = os.path.join(DATA_DIR, 'patches_test/*.npy')
patches_fp = glob.glob(pattern)

""" ! When activating/deactivating, change filenames below accordingly! 3x 
    (is being done automatically in the code below since update)"""

print(f"Inference on {len(patches_fp)} patches.")

total_size = sum(os.path.getsize(file) for file in patches_fp)
print(f"Total size of patches: {total_size / 1e9:.2f} GB")
## ****************************************************************************
norm_np = (0, 1)
## ****************************************************************************
SELECTED_CHANNELS = list(range(517, 717))   #sensor_ranges = {"RGB": range(0, 3),"NIR": range(3, 227),"IR": range(227, 517),"THz": range(517, 717),"EF": range(0, 717),}
LABELS = {0: "Wood", 1: "Non_wood"}
###############################################################################
file_path_mean_std = r"C:\Users\Daten Analyse\ASKIVIT_1_5_means_and_stds_channel_range_0_to_717.json"
###############################################################################
sys.path.append(os.path.join(CWD))
import torchaskivit.custom_transforms as ct
import torchaskivit.cnn_classes as cnn
import torchaskivit.evalutils as evalutils


""" Transformations. """
# Selects which stats to use:

with open(file_path_mean_std, "r") as json_file_means_stds:
    data = json.load(json_file_means_stds)

if stats == "ImageNet":
    means = torch.tensor([
        0.485,
        0.456,
        0.406
    ])
    stds = torch.tensor([
        0.229,
        0.224,
        0.225
    ])
elif stats == "ASKIVIT_V2.0":
    means = torch.tensor([
        0.37899301215195424,
        0.31209731802557417,
        0.2958513204345865
    ])
    stds = torch.tensor([
        0.3360531244381102,
        0.30271688736296726,
        0.3105229391608789
    ])
elif stats == "ASKIVIT_V1.4":
    means = torch.tensor([
        0.00065407,
        0.00084001,
        0.00102502
    ])
    stds = torch.tensor([
        0.01421314,
        0.01720814,
        0.0203272
    ])
elif stats == "ASKIVIT_V1.5":
    all_means = torch.tensor(data["means"])
    means = all_means[SELECTED_CHANNELS]
    all_stds = torch.tensor(data["stds"])
    stds = all_stds[SELECTED_CHANNELS]

transforms = v2.Compose([
    ct.SelectChannels(SELECTED_CHANNELS),
    ct.NormalizeNumpy(min_val=norm_np[0], max_val=norm_np[1]),
    # ct.BGR2RGB(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=False),
    # v2.Resize((300, 300), antialias=True),
    v2.Normalize(mean=means, std=stds)
])

""" Loss function. """
criterion = torch.nn.CrossEntropyLoss()

for model_cp_path in all_models:
    model_cp = os.path.basename(model_cp_path)
    print(f"Model: {model_cp}")


    """ Load model. """
    # Load model checkpoint.
    checkpoint = torch.load(model_cp_path)

    # Initialize model.
    cnn_class = getattr(cnn, choose_model, None)
    if cnn_class is None:
        raise AttributeError("Model not found in torchaskivit.cnn_classes module.")

    model = cnn_class(selected_channels=SELECTED_CHANNELS, input_size=50)
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set model to evaluation mode.
    model.eval()

    # Track correct predictions and total predictions for accuracy calculation.
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0

    y_pred = []
    y_true = []
    gt = []

    # Initialize dictionaries to store examples
    correct_examples = {"Wood": [], "Non_wood": []}
    incorrect_examples = {"Wood": [], "Non_wood": []}

    for file_path in tqdm.tqdm(patches_fp):
        # Load the patch.
        patch = np.load(file_path)

        #print(f"Original patch shape: {patch.shape}")

        patch_transformed = transforms(patch)

        # Add batch dimension.
        patch_transformed = patch_transformed.unsqueeze(0)  
        
        # Move patch to the same device as model.
        patch_transformed = patch_transformed.to(device)
        
        # Perform inference.
        with torch.no_grad():
            output = model(patch_transformed)
            predicted_label = output.argmax(dim=1, keepdim=True).cpu().item()

        # Get actual label.
        main_label_idx, label_idx = evalutils.get_actual_label_ASKIVIT_V1_5(file_path)

        # Compute loss.
        loss = criterion(output, torch.tensor([main_label_idx]).to(device))
        total_loss += loss.item()

        # Update correct_predictions, total_predictions, and categorize examples.
        predicted_label_name = LABELS[predicted_label]
        actual_label_name = LABELS[main_label_idx]  # This assumes actual_label is already the numerical label

        if predicted_label == main_label_idx:
            correct_predictions += 1
            correct_examples[actual_label_name].append((file_path, predicted_label_name))  # Storing the string name for consistency
        else:
            incorrect_examples[actual_label_name].append((file_path, predicted_label_name, actual_label_name))  # Include both predicted and actual labels
        total_predictions += 1

        # Store for confusion matrix.
        y_pred.append(predicted_label)
        y_true.append(main_label_idx)
        gt.append(label_idx)

    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {total_predictions}")
    # Calculate accuracy.
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Average loss.
    average_loss = total_loss / total_predictions
    print(f"Average loss: {average_loss:.4f}")

    # Add to model metadata.
    checkpoint["metadata"]["v1.4_acc"] = accuracy
    checkpoint["metadata"]["v1.4_loss"] = average_loss

    # Save model metadata.
    dir_name = os.path.dirname(model_cp_path)
    filename = "with_v1.5_test_result_" + model_cp
    new_model_cp = os.path.join(dir_name, filename)

    torch.save(checkpoint, new_model_cp)


    """ Store for confusion matrix. """
    predictions_filename = f"predictions_{model_cp[:13]}_tested"
    predictions_filepath = os.path.join(DIR_MODELS, predictions_filename)

    data = {
        "y_pred": y_pred,
        "y_true": y_true,
        "gt": gt,
    }

    with open(predictions_filepath+".json", 'w') as f:
        json.dump(data, f)

    cm = evalutils.generate_cm_correct_vs_uncorrect(data, mode="v1.5")
    cm_filename = f"cm_{model_cp[:13]}_with_acc_{accuracy}"
    cm_filepath = os.path.join(DIR_MODELS, cm_filename)
    np.savetxt(cm_filepath+".txt", cm, fmt="%d")

    # Plot confusion matrix.
    fig, ax = evalutils.plot_cm(cm, mode="v1.5")
    fig.savefig(cm_filepath+".pdf", bbox_inches="tight")

    #plt.show()