# -*- coding: utf-8 -*-
__author__ = "Felix Niemöller"
__email__ = "usnrb@student.kit.edu"

import torch

# torchaskivit.utils
from . import utils


def train_one_epoch(epoch, P, d, print_interval=50):
    len_train = d["loader_train_len"]

    # Initialize loss tracking
    running_loss = 0.0
    last_loss = 0.0
    total_loss = 0.0

    # Initialize accuracy tracking
    correct = 0
    total = 0

    for i, data in enumerate(d["loader_train"]):
        images, labels = data[0], data[1]
        images = images.to(P["DEVICE"], non_blocking=True)
        labels = labels.to(P["DEVICE"], non_blocking=True)

        d["optimizer"].zero_grad(set_to_none=True)
        last_lr = d["scheduler"].get_last_lr()[0]
        d["tb_writer"].add_scalar(
            "Learning Rate", last_lr, epoch * len_train + i)

        """ Forward Pass """
        # Predictions
        outputs = d["model"](images)

        # Get predictions and count correct ones
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        """ Backward Pass """
        # Compute loss and gradients
        loss = d["criterion"](outputs, labels)
        loss.backward()

        # Adjust weights
        d["optimizer"].step()
        d["scheduler"].step()

        """ Gather data and report """
        running_loss += loss.item()
        total_loss += loss.item()

        if i % print_interval == print_interval-1:
            last_loss = running_loss / print_interval  # Loss per batch
            utils.lp(f"   batch {i+1:4}/{len_train} | loss: {last_loss:.4f}", P)

            running_loss = 0.0
        
        if i == d["loader_train_len"] - 1:
            last_loss = running_loss / (i % print_interval + 1)
            utils.lp(f"   batch {i+1:4}/{len_train} | loss: {last_loss:.4f}", P)

    # Calculate weigthed total average loss
    avg_total_loss = total_loss / len_train

    # Calculate accuracy
    accuracy = 100 * correct / total

    d["tb_writer"].add_scalar("Loss/train", last_loss, epoch)
    d["tb_writer"].add_scalar("Accuracy/train", accuracy, epoch)

    return avg_total_loss, accuracy



def validate_model(epoch, P, d):
    # Initialize loss tracking
    running_val_loss = 0.0

    # Initialize accuracy tracking
    correct = 0
    total = 0

    for i, vdata in enumerate(d["loader_val"]):
        vimages, vlabels = vdata[0], vdata[1]
        vimages = vimages.to(P["DEVICE"], non_blocking=True)
        vlabels = vlabels.to(P["DEVICE"], non_blocking=True)

        voutputs = d["model"](vimages)

        _, predicted = torch.max(voutputs.data, 1)
        total += vlabels.size(0)
        correct += (predicted == vlabels).sum().item()

        vloss = d["criterion"](voutputs, vlabels)
        running_val_loss += vloss.item()

    # Calculate accuracy
    val_accuracy = 100 * correct / total

    d["tb_writer"].add_scalar("Loss/val", running_val_loss / (i + 1), epoch)
    d["tb_writer"].add_scalar("Accuracy/val", val_accuracy, epoch)

    return running_val_loss / (i + 1), val_accuracy


def test_model(epoch, P, d):
    # Initialize loss tracking
    running_test_loss = 0.0

    # Initialize accuracy tracking
    correct = 0
    total = 0

    for i, vdata in enumerate(d["loader_test"]):
        timages, tlabels = vdata[0], vdata[1]
        timages = timages.to(P["DEVICE"], non_blocking=True)
        tlabels = tlabels.to(P["DEVICE"], non_blocking=True)

        toutputs = d["model"](timages)

        # Check for NaN/Inf in outputs
        # Implemented because the were corrupted files found in the test set of
        # ASKIVIT V2.0 NIR, causing NaN/Inf in the outputs and problems.
        if torch.isnan(toutputs).any() or torch.isinf(toutputs).any():
            print(f"NaN/Inf detected in outputs at batch {i}")
            continue  # Skip this batch or handle as needed

        _, predicted = torch.max(toutputs.data, 1)
        total += tlabels.size(0)
        correct += (predicted == tlabels).sum().item()

        tloss = d["criterion"](toutputs, tlabels)
        
        # Check for NaN/Inf in loss
        if torch.isnan(tloss) or torch.isinf(tloss):
            print(f"NaN/Inf detected in loss at batch {i}")
            continue  # Skip this batch or handle as needed
        
        running_test_loss += tloss.item()

    # Calculate accuracy
    test_accuracy = 100 * correct / total

    d["tb_writer"].add_scalar("Loss/test", running_test_loss / (i + 1), epoch)
    d["tb_writer"].add_scalar("Accuracy/test", test_accuracy, epoch)

    return running_test_loss / (i + 1), test_accuracy



class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

        self.best_loss = float("inf")
        self.best_val_acc = 0
        self.best_epoch = 0

        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True