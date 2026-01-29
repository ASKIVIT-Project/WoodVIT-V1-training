# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import re
import json
import glob

DIR= os.path.abspath(r"C:\Users")
# if dir not in sys.path:
sys.path.append(DIR)
import torchaskivit.constants as constants


def get_actual_label_ASKIVIT_V2(filepath):
    """ Get actual label from filepath. """
    # Get filename without path.
    npy_filename = os.path.basename(filepath)
    
    # get label out of file name
    if "GT_" not in npy_filename:

        npy_filename_sliced = "_".join(npy_filename.split(".")[1].split("_")[3:])

        if "Non_wood_" in npy_filename_sliced:
            main_label = "_".join(npy_filename_sliced.split("_")[0:2])

            sub_label = "_".join(npy_filename_sliced.split("_")[2:])

        elif "Wood_" in npy_filename_sliced:
            main_label = "_".join(npy_filename_sliced.split("_")[0:1])

            sub_label = "_".join(npy_filename_sliced.split("_")[1:])

    else:
        raise ValueError("GT file handed to get_acutal_label_AKSIVIT_V2 instead of regular npy.")

    # Map label to two main classes.
    if main_label == "Wood":
        main_class_idx = 0
    elif main_label == "Non_wood":
        main_class_idx = 1
    else:
        raise ValueError(f"Label {main_label} not in class_to_idx")
    
    sub_label_idx = constants.ASKIVIT_LABELS_V2.get(sub_label)
    main_label_idx = constants.ASKIVIT_LABELS_V2_MAIN.get(main_label)

    return main_label_idx, sub_label_idx


def get_actual_label_ASKIVIT_V1(filepath):
    """ Get actual label from filepath. """
    filename = os.path.basename(filepath)

    label = "_".join(filename.split(".")[0].split("_")[-2:])
    main_label = label.split("_")[0]

    label_idx = constants.ASKIVIT_LABELS_V1.get(label)
    main_label_idx = constants.ASKIVIT_LABELS_V1_MAIN.get(main_label)

    return main_label_idx, label_idx


def get_actual_label_ASKIVIT_V1_5(filepath):
    """ Get actual label from filepath. """
    filename = os.path.basename(filepath)
    label = "_".join(filename.split(".")[0].split("_")[5:])

    if "Non" in label:
        main_label = "_".join(label.split("_")[0:2])

        sub_label = "_".join(label.split("_")[2:])

    elif "Wood" in label:
        main_label = "_".join(label.split("_")[0:1])

        sub_label = "_".join(label.split("_")[1:])
    else:
        Warning('IF error in get_actual_label_ASKIVIT_V1_5')
        main_label=None
        sub_label=None

    sub_label_idx = constants.ASKIVIT_LABELS_V1_5.get(sub_label)
    main_label_idx = constants.ASKIVIT_LABELS_V1_5_MAIN.get(main_label)
    
    return main_label_idx, sub_label_idx

def get_actual_label_ASKIVIT_V1_5_metal_expert(filepath):
    """ Get actual label from filepath. """
    filename = os.path.basename(filepath)
    label = "_".join(filename.split(".")[0].split("_")[5:])

    if "Non" in label:
        main_label = "_".join(label.split("_")[0:2])

        sub_label = "_".join(label.split("_")[2:])

    elif "Wood" in label:
        main_label = "_".join(label.split("_")[0:1])

        sub_label = "_".join(label.split("_")[1:])

    if "Metal" in sub_label:
        main_label = "Metal"
    else:
        main_label = "Non_metal"

    sub_label_idx = constants.ASKIVIT_LABELS_V1_5_METAL_EXPERT.get(sub_label)
    main_label_idx = constants.ASKIVIT_LABELS_V1_5_METAL_EXPERT_MAIN.get(main_label)
    
    return main_label_idx, sub_label_idx

def get_actual_label_ASKIVIT_V2_metal_expert(filepath):
    """ Get actual label from filepath. """
    # Get filename without path.
    npy_filename = os.path.basename(filepath)
    
    # get label out of file name
    if "GT_" not in npy_filename:

        npy_filename_sliced = "_".join(npy_filename.split(".")[1].split("_")[3:])

        if "Non_wood_" in npy_filename_sliced:
            main_label = "_".join(npy_filename_sliced.split("_")[0:2])

            sub_label = "_".join(npy_filename_sliced.split("_")[2:])

        elif "Wood_" in npy_filename_sliced:
            main_label = "_".join(npy_filename_sliced.split("_")[0:1])

            sub_label = "_".join(npy_filename_sliced.split("_")[1:])

    else:
        raise ValueError("GT file handed to get_acutal_label_AKSIVIT_V2 instead of regular npy.")

    # Map label to two main classes.
    if sub_label == "Metal":
        main_class_idx = 0
    elif "metal" in sub_label:
        main_class_idx = 0
    else:
        main_class_idx = 1
    
    sub_label_idx = constants.ASKIVIT_LABELS_V2_METAL_EXPERT.get(sub_label)
    #main_label_idx = constants.ASKIVIT_LABELS_V2_MAIN_METAL_EXPERT.get(main_label)

    return main_class_idx, sub_label_idx

def generate_prediction_dict(inference: dict) -> dict:
    """
    Build prediction data from an inference JSON for downstream evaluation.
    Supports wood ('Non_wood'|'Wood') and metal expert ('Non_metal'|'Metal').

    Returns
    -------
    data : dict
        {
          'y_pred': list[int],  # binary predictions  (Non_* -> 1, * -> 0)
          'y_true': list[int],  # binary GT (Non_* -> 1, * -> 0)
          'gt':     list[int],  # multi-class GT index from ASKIVIT_LABELS_V1_5 via SUBCLASS
        }
    """
    results = inference.get("results", [])
    if not isinstance(results, list):
        raise ValueError("inference['results'] must be a list")
    if not results:
        return {"y_pred": [], "y_true": [], "gt": []}

    wood_set  = {"Non_wood", "Wood"}
    metal_set = {"Non_metal", "Metal"}

    # Prefer prediction to detect domain; fallback to filename marker.
    def _detect_domain(item0):
        pred = item0.get("predicted_label_name")
        if pred in metal_set:
            return "metal"
        if pred in wood_set:
            return "wood"
        stem = os.path.splitext(os.path.basename(item0.get("patch", "")))[0]
        m = re.search(r"_(Non_wood|Wood|Non_metal|Metal)_(.+)$", stem)
        mc = m.group(1) if m else None
        if mc in metal_set:
            return "metal"
        if mc in wood_set:
            return "wood"
        raise ValueError("Cannot detect domain (wood/metal).")

    domain = _detect_domain(results[0])

    # Encoding convention (your latest): Non_* -> 1, * -> 0
    if domain == "wood":
        pred_label_map      = {"Non_wood": 1, "Wood": 0}
        filename_mainclass  = {"Non_wood", "Wood"}
        tail_regex          = re.compile(r"_(Non_wood|Wood)_(.+)$")
    else:  # metal
        pred_label_map      = {"Non_metal": 1, "Metal": 0}
        filename_mainclass  = {"Non_metal", "Metal"}
        tail_regex          = re.compile(r"_(Non_metal|Metal|Non_wood|Wood)_(.+)$")  # allow wood MAINCLASS in filenames

    # For metal domain, derive GT Non_metal/Metal from SUBCLASS if filename MAINCLASS is wood-like.
    metal_subclasses = {
        "Metal", "Metal_covered_by_Wood", "Metal_covered_by_Upholstery", "Metal_covered_by_Plastic"
    }
    # Everything else (Background, Plastic, Mineralics, Upholstery, Wood_covered_by_*) -> Non_metal

    def _parse_tail(patch_path: str):
        stem = os.path.splitext(os.path.basename(patch_path))[0]
        m = tail_regex.search(stem)
        if not m:
            raise ValueError(
                f"Cannot extract MAINCLASS/SUBCLASS from filename: {os.path.basename(patch_path)}. "
                "Expected ..._{MAINCLASS}_{SUBCLASS}.npy"
            )
        return m.group(1), m.group(2)  # MAINCLASS token found, SUBCLASS (may contain underscores)

    y_pred, y_true, gt = [], [], []

    for i, item in enumerate(results):
        # --- y_pred from predicted label ---
        pred_name = item.get("predicted_label_name")
        if pred_name not in pred_label_map:
            raise ValueError(
                f"[item {i}] Unknown predicted label {pred_name!r} for domain '{domain}'. "
                f"Expected one of {list(pred_label_map)}."
            )
        y_pred.append(pred_label_map[pred_name])

        # --- parse filename tail ---
        patch = item.get("patch")
        if patch is None:
            raise ValueError(f"[item {i}] Missing 'patch' in inference item.")
        mainclass_token, subclass = _parse_tail(patch)

        # --- y_true (binary) ---
        if domain == "wood":
            # MAINCLASS is expected to be Non_wood|Wood
            if mainclass_token not in filename_mainclass:
                raise ValueError(f"[item {i}] MAINCLASS {mainclass_token!r} not valid for wood domain.")
            y_true.append(1 if mainclass_token.startswith("Non_") else 0)

        else:  # domain == "metal"
            # If filename has metal-style MAINCLASS, use it; if it has wood-style, infer from SUBCLASS.
            if mainclass_token in {"Non_metal", "Metal"}:
                y_true.append(1 if mainclass_token == "Non_metal" else 0)
            elif mainclass_token in {"Non_wood", "Wood"}:
                # Infer from SUBCLASS membership
                is_metal = subclass in metal_subclasses
                y_true.append(0 if is_metal else 1)  # Non_metal->1, Metal->0
            else:
                raise ValueError(f"[item {i}] Unexpected MAINCLASS {mainclass_token!r} in metal domain.")

        # --- gt (multiclass) from SUBCLASS table ---
        gt_val = constants.ASKIVIT_LABELS_V1_5.get(subclass)
        if gt_val is None:
            raise ValueError(
                f"[item {i}] Unknown SUBCLASS/GT label {subclass!r} "
                "(not found in ASKIVIT_LABELS_V1_5)."
            )
        gt.append(gt_val)

    return {"y_pred": y_pred, "y_true": y_true, "gt": gt}


# Optional quick test harness
def TEST_generate_prediction_dict():
    print("TESTING: generate_prediction_dict()")
    # RGB
    # THz
    inference_path = r"D:\Askivit 1.5 2ter Datensatz mit korrigierter Datamap\Metall Experte\Varianten\Nur THz Bereich\Inference\Test Patches\metal_expert_inference_results.json"
    prediction_path = r"D:\Askivit 1.5 2ter Datensatz mit korrigierter Datamap\Metall Experte\Varianten\Nur THz Bereich\Inference\Test Patches\metal_expert_prediction.json.json"
    with open(inference_path, "r", encoding="utf-8") as f:
        inference = json.load(f)

    data = generate_prediction_dict(inference)

    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("Saved predictions.json")


def generate_cm_correct_vs_uncorrect(data: dict, mode="v1.5") -> np.ndarray:
    """
    NOT CORRECT CM but usefull: Plots correct classified vs not_correct
    (no matter how the binding between subclass and main class actually is defined)
    For THz-ME can be hard to read

    Generates a confusion matrix based on predictions and ground truth labels.


    Parameters
    ----------
    data: dict
        A dictionary containing the following key-value pairs:
        'y_pred': list or array-like
            The predicted labels.
        'y_true': list or array-like
            The true labels.
        'gt': list or array-like
            The ground truth categories that determine if a sample is
            considered 'wood' or 'non-wood'.
    mode: str, optional
        The operational mode that determines the number of classes
        and the categorization criteria. Should be either 'v1' or 'v2'.
        The default is 'v1'.

    Returns
    -------
    np.ndarray
        A 2D NumPy array representing the confusion matrix, where
        the first dimension is 'wood' vs. 'non-wood', and the
        second dimension represents the classes corresponding to
        the mode.
    """

    if mode not in ["v1", "v1.5", "v2"]:
        raise ValueError("Mode must be either 'v1' or 'v1.5' or 'v2'.")

    mode_num_classes = {"v1": 16, "v1.5": 16, "v2": 14}
    mode_wood_idxs = {"v1": [0, 1, 2, 3],"v1.5": [0, 1, 2, 3], "v2": [0, 1, 2, 3]}

    num_classes = mode_num_classes.get(mode)
    wood_idxs = mode_wood_idxs.get(mode)

    cm = np.zeros((2, num_classes))
    eval_results = np.equal(data["y_pred"], data["y_true"])

    for i in range(len(eval_results)):
        row = 0 if data["gt"][i] in wood_idxs else 1

        if eval_results[i]:
            cm[row, data["gt"][i]] += 1
        else:
            cm[int(np.abs(row - 1)), data["gt"][i]] += 1

    return cm

import numpy as np

def generate_cm(data):
    """
    Generate a 2×N confusion matrix for binary main-class predictions
    versus subclass ground truth.

    Rows:
        0 → predicted 'Wood' (or 'Metal')
        1 → predicted 'Non_*'

    Columns:
        subclass indices (0 .. N-1) taken from data['gt'].

    Parameters
    ----------
    data : dict
        Must contain:
            - 'y_pred': list[int]  (binary predictions, 0/1)
            - 'gt'    : list[int]  (ground-truth subclass indices)

    Returns
    -------
    cm : np.ndarray of shape (2, num_classes)
        Confusion matrix with predicted class (rows) vs subclass index (columns).
    """
    y_pred = np.asarray(data["y_pred"], dtype=int)
    gt = np.asarray(data["gt"], dtype=int)

    # --- Sanity checks ---
    if len(y_pred) != len(gt):
        raise ValueError("y_pred and gt must have equal length.")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError(f"y_pred must contain only 0/1 values, found {np.unique(y_pred)}.")
    if gt.size == 0:
        raise ValueError("gt array is empty.")

    num_classes = int(np.max(gt)) + 1
    cm = np.zeros((2, num_classes), dtype=int)

    # --- Count occurrences for each (prediction, subclass) pair ---
    for yp, g in zip(y_pred, gt):
        cm[int(yp), int(g)] += 1

    return cm

def plot_cm_recall_dual(cm, mode="v1.5"):
    """
    Generalized confusion matrix visualization for ASKIVIT subclasses.

    Features:
    ---------------------------------------------------------------------
    - Plots subclass confusion matrix (2×N) with recall row below.
    - Adds a 2×2 main-class confusion matrix (right) without color mapping.
    - Uses a unified red–yellow–green colormap for subclass visualization.
    - Displays metrics for the main-class matrix (class_0 as positive).

    Interpretation:
      class_0 → the "main" class (e.g. Wood or Metal)
      class_1 → the complement (Non-Wood or Non-Metal)

    Confusion matrix cell meaning (Pred on Y, True on X):
          | True 0 | True 1 |
      ----+---------+--------+
      Pred 0 | TP=recall | FP=fpr
      Pred 1 | FN=fnr    | TN=specificity
    """
    matplotlib.pyplot.close() # Not tested

    # --- Style ---
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    small_fontsize = 11

    # --- Mode configuration ---
    if mode not in ["v1.5", "v1.5ME", "v2"]:
        raise ValueError("Mode must be 'v1.5', 'v1.5ME', or 'v2'.")

    labels = {
        "v1.5": constants.ASKIVIT_LABELS_V1_5_PLOT,
        "v1.5ME": constants.ASKIVIT_LABELS_V1_5_PLOT,  # placeholder for Metal Expert
        "v2": constants.ASKIVIT_LABELS_V2,
    }
    label_list = labels[mode]
    n_cols = len(label_list)

    # --- Subclass → main-class mapping ---
    if mode == "v1.5":
        subclass_to_main = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_labels = ["Predicted as\n Wood", "Predicted as\n Non-Wood"]
        x_labels = ["True Wood", "True Non-Wood"]
    elif mode == "v2":
        subclass_to_main = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_labels = ["Predicted as\n Wood", "Predicted as\n Non-Wood"]
        x_labels = ["True Wood", "True Non-Wood"]
    elif mode == "v1.5ME":
        subclass_to_main = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        y_labels = ["Predicted as\n Metal", "Predicted as\n Non-Metal"]
        x_labels = ["True\n Metal", "True\n Non-Metal"]

    # --- Compute main-class confusion matrix & metrics ---
    metrics_array, metrics_dict = compute_main_class_metrics_from_cm(cm, subclass_to_main)

    # --- Colormap (red→yellow→green) ---
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "red_yellow_green", ["#d7191c", "#f4e071", "#00876C"], N=256
    )

    # === Figure layout ===
    fig = plt.figure(figsize=(10, 3.2))
    left, width_sub, width_main, left_gap = 0.13, 0.60, 0.075, 0.10
    ax_cm = fig.add_axes([left, 0.57, width_sub, 0.52])
    eps = 1.0 / (fig.get_dpi() * fig.get_size_inches()[1])
    rec_h = 0.28 * ax_cm.get_position().height
    ax_rec = fig.add_axes([
        ax_cm.get_position().x0,
        ax_cm.get_position().y0 - rec_h - eps,
        ax_cm.get_position().width,
        rec_h
    ])
    ax_main = fig.add_axes([left + width_sub + left_gap, 0.57, width_main, 0.52])

    # -------------------------------------------------------------------------
    # Subclass Confusion Matrix
    # -------------------------------------------------------------------------
    colsum = np.sum(cm, axis=0)
    colsum[colsum == 0] = 1

    cm_norm = cm / colsum
    mask = np.array([[subclass_to_main[j] != i for j in range(n_cols)] for i in range(2)])
    cm_masked = np.ma.masked_where(mask, cm_norm)

    cmap_with_white = cmap.copy()
    cmap_with_white.set_bad(color="white")

    ax_cm.matshow(cm_masked, cmap=cmap_with_white, vmin=0, vmax=1)

    # Axes and grid
    ax_cm.set_xticks(np.arange(n_cols))
    ax_cm.set_xticklabels([])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_yticklabels(y_labels, fontsize=small_fontsize, va="center", ha="right")
    ax_cm.xaxis.set_ticks_position("bottom")
    ax_cm.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax_cm.set_yticks(np.arange(3) - 0.5, minor=True)
    ax_cm.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax_cm.axhline(y=0.5, color="black", linewidth=3)

    # Annotate absolute counts
    for i in range(2):
        for j in range(n_cols):
            ax_cm.text(j, i, f"{int(cm[i, j])}",
                       ha="center", va="center", fontsize=small_fontsize)

    # -------------------------------------------------------------------------
    # Recall Row
    # -------------------------------------------------------------------------
    correct_row = subclass_to_main.copy()
    recalls = cm[correct_row, np.arange(n_cols)] / colsum
    recalls = np.clip(recalls, 0, 1)
    rec_matrix = np.expand_dims(recalls, axis=0)
    ax_rec.matshow(rec_matrix, cmap=cmap, vmin=0, vmax=1)

    ax_rec.set_xticks(np.arange(n_cols))
    ax_rec.set_xticklabels(label_list, rotation=45, ha="right", fontsize=small_fontsize)
    ax_rec.set_yticks([0])
    ax_rec.set_yticklabels(["Recall"], fontsize=small_fontsize)
    ax_rec.xaxis.set_ticks_position("bottom")
    ax_rec.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax_rec.set_yticks(np.arange(2) - 0.5, minor=True)
    ax_rec.grid(which="minor", color="black", linestyle="-", linewidth=1)

    # Recall text annotations (0.xx)
    for j in range(n_cols):
        ax_rec.text(j, 0, f"{recalls[j]:.2f}",
                    ha="center", va="center", fontsize=small_fontsize)

    # -------------------------------------------------------------------------
    # Main-Class Matrix (2×2, white background + metrics text)
    # -------------------------------------------------------------------------

    # White background fill
    bg = np.ones((2, 2, 3), dtype=float)
    ax_main.imshow(bg, extent=(-0.5, 1.5, 1.5, -0.5), zorder=0)
    ax_main.set_xlim(-0.5, 1.5)
    ax_main.set_ylim(1.5, -0.5)
    ax_main.set_aspect("equal", adjustable="box")

    # Axis labels
    ax_main.set_xticks([0, 1])
    ax_main.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=small_fontsize)
    ax_main.set_yticks([0, 1])
    ax_main.set_yticklabels(y_labels, fontsize=small_fontsize, va="center", ha="right")
    ax_main.xaxis.set_ticks_position("bottom")

    # Grid lines (thin)
    ax_main.set_xticks([-0.5, 0.5, 1.5], minor=True)
    ax_main.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax_main.grid(which="minor", color="black", linestyle="-", linewidth=1)

    for spine in ax_main.spines.values():
        spine.set_visible(False)

    # --- Metrics text (class_0 as positive reference) ---
    m = metrics_dict["class_0"]
    ax_main.text(0, 0, f"{m['recall']:.2f}",       ha="center", va="center", fontsize=small_fontsize)
    ax_main.text(1, 0, f"{m['fpr']:.2f}",          ha="center", va="center", fontsize=small_fontsize)
    ax_main.text(0, 1, f"{m['fnr']:.2f}",          ha="center", va="center", fontsize=small_fontsize)
    ax_main.text(1, 1, f"{m['specificity']:.2f}",  ha="center", va="center", fontsize=small_fontsize)

    return fig, (ax_cm, ax_rec, ax_main)

def compute_main_class_metrics_from_cm(
    cm: np.ndarray,
    subclass_to_main: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """
    Compute 2×2 main-class confusion matrix and full set of metrics for both
    main classes (each considered as the positive class once).

    Args:
        cm (np.ndarray): 2×N confusion matrix: Where rows = true main classes,
                         columns = predicted subclasses.
                         I repeat we use intentionally a swapped confusion matrix with rows: true main classes
                         and columns = predictions
        subclass_to_main (np.ndarray, optional): length-N array assigning each
                         subclass column to its main class (0 or 1).
                         Defaults to ASKIVIT v1.5 layout:
                         [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1].

    Returns:
        tuple:
            metrics_array (np.ndarray): shape (2, 9)
                columns = [
                    recall, fnr, fpr, specificity, f1, acc,
                    recall_macro, f1_macro, acc_macro
                ]
                row 0: class 0 positive
                row 1: class 1 positive

            metrics_dict (dict): keyworded dictionary with:
                {
                    "cm_main": 2×2 confusion matrix,
                    "class_0": {...metrics...},
                    "class_1": {...metrics...},
                    "macro": {...macro_averages...}
                }
    """
    # --- Default subclass-to-main mapping (v1.5 Wood/Non-Wood) ---
    if subclass_to_main is None:
        subclass_to_main = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)
        Warning("Using default subclass to main class in compute_main_class_metrics_from_cm")

    # class 0 is Wood, class 1 is Non-Wood
    # class 0 is This, class 1 is Non-this

    n_cols = cm.shape[1]
    if len(subclass_to_main) != n_cols:
        raise ValueError(
            f"Length of subclass_to_main ({len(subclass_to_main)}) "
            f"must match cm columns ({n_cols})."
        )

    # --- Aggregate subclass confusion matrix into main-class (2×2) ---
    cm_main = np.zeros((2, 2), dtype=float)
    for j in range(n_cols):
        pred_main = subclass_to_main[j]
        cm_main[0, pred_main] += cm[0, j]
        cm_main[1, pred_main] += cm[1, j]

    eps = 1e-12
    metrics_array = np.zeros((2, 13), dtype=float)  # rows = classes, cols = metrics

    # helper for MCC
    def _mcc(tp, tn, fp, fn):
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return ((tp * tn) - (fp * fn)) / (np.sqrt(denom) + eps)

    def _mcc_norm(tp, tn, fp, fn):
        normMCC = (_mcc(tp, tn, fp, fn)+1)/2
        return normMCC

    # --- Compute metrics for both main classes explicitly ---
    per_class = {}

    # class 0 positive
    tp0 = cm_main[0, 0]
    fp0 = cm_main[0, 1]
    fn0 = cm_main[1, 0]
    tn0 = cm_main[1, 1]
    total0 = tp0 + tn0 + fp0 + fn0

    recall0 = tp0 / (tp0 + fn0 + eps)
    fnr0 = fn0 / (tp0 + fn0 + eps)
    fpr0 = fp0 / (fp0 + tn0 + eps)
    specificity0 = tn0 / (fp0 + tn0 + eps)
    precision0 = tp0 / (tp0 + fp0 + eps)
    f10 = 2 * (precision0 * recall0) / (precision0 + recall0 + eps)
    acc0 = (tp0 + tn0) / (total0 + eps)
    mcc0         = _mcc(tp0, tn0, fp0, fn0)
    norm_mcc0 = _mcc_norm(tp0, tn0, fp0, fn0)

    metrics_array[0, 0:8] = [
        recall0, fnr0, fpr0, specificity0, f10, acc0, mcc0,norm_mcc0
    ]

    per_class["class_0"] = {
        "tp": tp0,
        "fp": fp0,
        "fn": fn0,
        "tn": tn0,
        "recall": recall0,
        "fnr": fnr0,
        "fpr": fpr0,
        "specificity": specificity0,
        "precision": precision0,
        "f1": f10,
        "acc": acc0,
        "mcc": mcc0,
        "norm_mcc": norm_mcc0,
    }

    # === Class 1 as positive (e.g. Non-Wood or Non-Metal) ===
    tp1 = cm_main[1, 1]
    fp1 = cm_main[1, 0]
    fn1 = cm_main[0, 1]
    tn1 = cm_main[0, 0]
    total1 = tp1 + tn1 + fp1 + fn1

    recall1 = tp1 / (tp1 + fn1 + eps)
    fnr1 = fn1 / (tp1 + fn1 + eps)
    fpr1 = fp1 / (fp1 + tn1 + eps)
    specificity1 = tn1 / (fp1 + tn1 + eps)
    precision1 = tp1 / (tp1 + fp1 + eps)
    f11 = 2 * (precision1 * recall1) / (precision1 + recall1 + eps)
    acc1 = (tp1 + tn1) / (total1 + eps)
    mcc1         = _mcc(tp1, tn1, fp1, fn1)
    norm_mcc1 = _mcc_norm(tp1, tn1, fp1, fn1)

    metrics_array[1, 0:8] = [
        recall1, fnr1, fpr1, specificity1, f11, acc1, mcc1, norm_mcc1
    ]
    per_class["class_1"] = {
        "tp": tp1,
        "fp": fp1,
        "fn": fn1,
        "tn": tn1,
        "recall": recall1,
        "fnr": fnr1,
        "fpr": fpr1,
        "specificity": specificity1,
        "precision": precision1,
        "f1": f11,
        "acc": acc1,
        "mcc": mcc1,
        "norm_mcc": norm_mcc1,
    }

    # --- Macro-averaged metrics ---
    recall_macro = np.mean(metrics_array[:, 0])
    f1_macro = np.mean(metrics_array[:, 4])
    acc_macro = np.mean(metrics_array[:, 5])
    mcc_macro = metrics_array[:, 6].mean().item()
    norm_mcc_macro = metrics_array[:, 7].mean().item()
    metrics_array[:, 8:] = [recall_macro, f1_macro, acc_macro, mcc_macro, norm_mcc_macro]

    macro_dict = {
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "acc_macro": acc_macro,
        "mcc_macro": mcc_macro,
        "norm_mcc_macro": norm_mcc_macro,
    }

    # --- Build structured dictionary output ---
    metrics_dict = {
        "class_0": per_class["class_0"],
        "class_1": per_class["class_1"],
        "macro": macro_dict,
    }
    return metrics_array, metrics_dict


def save_metric_array_to_csv(array: np.ndarray, filename: str):
    """
    Save a 2×9 metrics array as a German-style CSV file:
      - Semicolon-separated (;)
      - Comma as decimal separator (,)
      - UTF-8 encoded
      - No comment symbols

    Columns: recall; fnr; fpr; specificity; f1; acc;
             recall_macro; f1_macro; acc_macro
    """
    header = "recall;fnr;fpr;specificity;f1;acc;recall_macro;f1_macro;acc_macro"

    # 1. Save with dot decimal (temporary text)
    temp_str = np.array2string(
        array,
        formatter={"float_kind": lambda x: f"{x:.6f}"},
        separator=";",
    )

    # 2. Clean up the numpy string output
    temp_str = temp_str.replace("[[", "").replace("]]", "")
    temp_str = temp_str.replace("[", "").replace("]", "").strip()

    # 3. Replace dots with commas for decimal separator
    temp_str = temp_str.replace(".", ",")

    # 4. Split rows and rebuild CSV text
    rows = temp_str.split("\n")
    csv_text = header + "\n" + "\n".join(rows)

    # 5. Write to file (always .csv)
    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(csv_text)

    print(f"Metrics saved (German CSV): {filename}")

def _detect_json_kind(payload: dict) -> str:
    """
    Returns 'inference' if payload looks like an inference JSON (has 'results'),
    'predictions' if it looks like the simplified dict (has 'y_pred' & 'y_true'),
    otherwise raises.
    """
    if isinstance(payload, dict) and "results" in payload:
        return "inference"
    if isinstance(payload, dict) and all(k in payload for k in ("y_pred", "y_true", "gt")):
        return "predictions"
    raise ValueError("Unknown JSON structure: expected inference{'results':[...]}"
                     " or predictions{'y_pred','y_true','gt'}.")


def full_evaluate_on_json(json_filepath: str,
                          save_path: str | None = None,
                          write_intermediate_predictions: bool = True,
                          mode = "v1.5",
                          silent = False):
    """
    Evaluate either an inference JSON (converts to predictions) or a predictions JSON.

    Parameters
    ----------
    json_filepath : str
        Path to either:
          - ...*_inference_results.json  (with key 'results'), or
          - ...*_prediction.json         (with keys 'y_pred','y_true','gt')
    save_path : str | None
        Folder to store outputs. Defaults to the parent folder of json_filepath.
    write_intermediate_predictions : bool
        If True and input is inference JSON, writes the derived predictions JSON next to outputs.
    mode : str

    Returns
    -------
    cm: Confusion matrix with absolut values cm_metrics
    metrics_array: different metrics per relevant class: in form class0: row0 and class1: row1
    metric_dict: ALL metric values as a large dict


    """


    # ---- Load file ----
    with open(json_filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    kind = _detect_json_kind(payload)

    # ---- Convert to 'data' dict if needed ----
    if kind == "inference":
        data = generate_prediction_dict(payload)  # uses your previously defined function
        base_name = os.path.splitext(os.path.basename(json_filepath))[0]
        pred_filename = f"{base_name}_prediction.json"
    else:  # 'predictions'
        data = payload
        base_name = os.path.splitext(os.path.basename(json_filepath))[0]
        pred_filename = os.path.basename(json_filepath)  # already a predictions file

    # ---- Save dir ----
    save_dir = save_path if save_path is not None else os.path.dirname(json_filepath)
    os.makedirs(save_dir, exist_ok=True)

    # ---- Optionally write the (new) predictions JSON ----
    if kind == "inference" and write_intermediate_predictions:
        pred_out_path = os.path.join(save_dir, pred_filename)
        with open(pred_out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Derived predictions written to: {pred_out_path}")

    # ---- Sanity checks ----
    n = len(data["y_pred"])
    if not (len(data["y_true"]) == len(data["gt"]) == n):
        raise ValueError(f"Length mismatch: y_pred={len(data['y_pred'])}, "
                         f"y_true={len(data['y_true'])}, gt={len(data['gt'])}")

    # ---- Compute CM + metrics ----
    cm = generate_cm(data)
    if mode =="v1.5ME":
        subclass_to_main = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        metrics_array, metrics_dict = compute_main_class_metrics_from_cm(cm, subclass_to_main)
    else:
        metrics_array, metrics_dict = compute_main_class_metrics_from_cm(cm)

    if not silent:
        print("=== Main-Class Confusion Matrix ===")
        cm_main_arr = np.array([
            [metrics_dict["class_0"]["tp"], metrics_dict["class_0"]["fp"]],
            [metrics_dict["class_0"]["fn"], metrics_dict["class_0"]["tn"]],
        ])
        print(cm_main_arr)
        print("\n=== Metrics Array (rounded) ===")
        print(np.round(metrics_array, 3))

    # ---- Save CM ----
    cm_filename="cm"
    cm_filepath = os.path.join(save_dir, cm_filename)
    np.savetxt(cm_filepath+".txt", cm, fmt="%d")

    # ---- Save metrics ----
    metrics_path = os.path.join(save_dir, "cm_metrics")
    save_metric_array_to_csv(metrics_array, metrics_path)

    # ---- Plot & save CM ----
    plot_cm_recall_dual(cm, mode=mode)  # assumes it draws on current figure

    parent_folder = os.path.basename(os.path.dirname(json_filepath))
    base_filename = f"cm_recall_dual_{parent_folder}_{base_name}"

    pdf_path = os.path.join(save_dir, base_filename + ".pdf")
    png_path = os.path.join(save_dir, base_filename + ".png")

    # Save both formats
    plt.savefig(pdf_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=600, bbox_inches="tight")

    if not silent:
        plt.show()  # or plt.close() if generating multiple figures
        print(f"Saved metrics to: {metrics_path}")
        print(f"Saved confusion matrix PDF to: {pdf_path}")

    return cm, metrics_array, metrics_dict

def TEST_full_evaluate_on_json():
    mode = "v1.5"
    #RGB test
    inference_path = r"D:\GIT_tree\Askivit 1.5 2ter Datensatz mit korrigierter Datamap\Vier Sensor Modelle\Models\RGB\predictions_250407_163150_tested.json"

    full_evaluate_on_json(inference_path,mode=mode)

def run_evaluations_on_4S_for_one_run(parent_path, mode=None):
    """
    For each expected model subfolder (EF/IR/NIR/RGB/THz):
      - find the newest 'predictions*.json' file
      - call full_evaluate_on_json on it (if found)

    Parameters
    ----------
    parent_path : str
        Path containing the subdirectories EF, IR, NIR, RGB, THz.
    mode : optional
        Optional argument forwarded to full_evaluate_on_json.
    """
    subfolder_names = ["EF", "IR", "NIR", "RGB", "THz"]
    subdirs = [os.path.join(parent_path, n) for n in subfolder_names]

    for d in subdirs:
        if not os.path.isdir(d):
            print(f"[SKIP] Not a directory or missing: {d}")
            continue

        # find predictions*.json; pick the most recently modified
        candidates = glob.glob(os.path.join(d, "predictions*.json"))
        if not candidates:
            print(f"[SKIP] No predictions*.json found in: {d}")
            continue

        newest = max(candidates, key=os.path.getmtime)
        print(f"[RUN ] Evaluating predictions file: {newest}")
        full_evaluate_on_json(newest, mode=mode)

    return subdirs


def Batch_run_evaluations_on_4S_for_one_run():
    """
    Use THIS to run the base modells. You have to change the 1ter bis 4ter path on your own and run it 4 times to create
    all plots for the base models

    Wrapper for run_evaluations_on_4S_for_one_run
    using the Askivit 1.5 dataset results path.
    """
    base_path = r"D:\GIT_tree\ba_wagner\06_results\Askivit 1.5 4ter Datensatz mit korrigierter Datamap"
    parent_path = os.path.join(base_path, "Vier Sensor Modelle", "Models")
    Mode=r'v1.5'

    print(f"[TEST] Running evaluations in:\n{parent_path}\n")
    run_evaluations_on_4S_for_one_run(parent_path, mode=Mode)

if __name__ == "__main__":
    #TEST_generate_prediction_dict()
    #TEST_full_evaluate_on_json()
    #TEST_run_evaluations_on_4S_for_one_run()
    Batch_run_evaluations_on_4S_for_one_run()
