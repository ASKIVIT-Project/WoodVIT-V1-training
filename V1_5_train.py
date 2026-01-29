import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

""" IMPORT CONFIG """
import V1_5_train_cfg

# cfg appends working directory to sys.path, so torchaskivit can be imported.
import torchaskivit.utils as utils
import torchaskivit.trainutils as trainutils
from torchaskivit.dataset import ASKIVIT_V1_5
from torch.optim.lr_scheduler import LambdaLR # , OneCycleLR



## ****************************************************************************
## MAIN                                                                       *
## ****************************************************************************

def main(P, d, start_time):
    # Get optional command line arguments. If not specified, use default values
    # from config file.
    args = utils.parse_args(P)


    """ Update Channels if specified in command line. """
    # If not, use channels specificed in config file.
    P["SELECTED_CH"] = utils.get_ch_for_sensor(args.sensor)
    P["SENSOR"] = args.sensor


    """ Choose model. """
    model = utils.choose_model(P, args.model)

    # Check if more than one GPU is available and wrap the model using DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(P["DEVICE"])
    model.apply(utils.initialize_weights) # Apply Xavier initialization
    assert next(model.parameters()).is_cuda
    model_name = model.__class__.__name__


    """ Update Statistics to adjust slicing of Statistics. """
    P["SLICED_MEANS"] = torch.tensor([P['MEANS'][i] for i in P['SELECTED_CH']])
    P["SLICED_STDS"] = torch.tensor([P['STDS'][i] for i in P['SELECTED_CH']])

    P["STATISTICS"]["ASKIVIT_V1.5"] = {
        "MEANS": P["SLICED_MEANS"],
        "STDS": P["SLICED_STDS"]
    }


    """ Create transform pipelines. """
    transform_train = utils.transform_pipeline(P, model_name, is_train=True, rgb=False) # benchmark
    transform_val = utils.transform_pipeline(P, model_name, is_train=False, rgb=False) # benchmark
    
    transform_train = v2.Compose(transform_train)
    transform_val = v2.Compose(transform_val)
    
    P["TRANSFORM_TRAIN"] = transform_train


    

    path = f"{model_name}_{P['SENSOR']}_{P['START_TIME_STR']}_tb_summary"
    tb_writer = SummaryWriter(os.path.join(P["TB_WRITER"], path))

    dataset_train = ASKIVIT_V1_5(
        P["DIR_TRAINSET"],
        class_to_idx=P["LABEL_IDX"],
        transform=transform_train)
    dataset_val = ASKIVIT_V1_5(
        P["DIR_VALSET"],
        class_to_idx=P["LABEL_IDX"],
        transform=transform_val)
    dataset_test = ASKIVIT_V1_5(
        P["DIR_TESTSET"],
        class_to_idx=P["LABEL_IDX"],
        transform=transform_val)

    loader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=P["BATCH_SIZE"],
        num_workers=P["NUM_WORKERS"],
        pin_memory=P["PIN_MEMORY"])
    loader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=P["BATCH_SIZE"],
        num_workers=P["NUM_WORKERS"],)
    loader_test = DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=P["BATCH_SIZE"],
        num_workers=P["NUM_WORKERS"],)
    


    # Add new Parameters to params dict.
    d.update({
        "tb_writer": tb_writer,
        "loader_train": loader_train,
        "loader_val": loader_val,
        "loader_test": loader_test,
        "model": model,
        "criterion": nn.CrossEntropyLoss(weight=P["LOSS_WEIGHTS"]),
        "optimizer": optim.Adam(model.parameters(), lr=args.lr), # benchmark
        "loader_train_len": len(loader_train),
    })
    
    # scheduler = OneCycleLR(
    #     d["optimizer"], 
    #     max_lr=0.01, 
    #     total_steps=d["loader_train_len"] * P["NUM_EPOCHS"], 
    #     pct_start=0.1, 
    #     anneal_strategy='cos', 
    #     div_factor=500.0, 
    #     final_div_factor=10000.0)

    lr_lambda = lambda epoch: 1
    scheduler = LambdaLR(
        d["optimizer"],
        lr_lambda=lr_lambda,
    )

    # Set up early stopping.
    early_stopping = trainutils.EarlyStopper(
        patience=P["PATIENCE"],
        min_delta=P["MIN_DELTA"])
    
    d.update({
        "scheduler": scheduler, 
        "early_stopping": early_stopping
    })

    utils.lp(utils.print_log(P, d, args), P)

    

    # Training Loop
    for epoch in range(P["NUM_EPOCHS"]):
        # Track learning rate.
        lastlr = d["scheduler"].get_last_lr()[0]

        # Print epoch
        utils.lp(f"\nEPOCH {epoch+1} | LR={lastlr}", P)
        

        """ Training step. """
        model.train() # Training mode.
        train_loss, train_acc = trainutils.train_one_epoch(epoch, P, d)


        """ Validation step. """
        model.eval() # Evaluation mode.
        with torch.no_grad():
            val_loss, val_acc = trainutils.validate_model(epoch, P, d)

        utils.lp(f"LOSS train: {train_loss:.4f}, val: {val_loss:.4f}", P)
        utils.lp(f" ACC train: {train_acc:.1f}%, val: {val_acc:.1f}%", P)
        utils.lp(f"t = {utils.get_time_elapsed(start_time)}", P)


        """ Saving best model. """
        early_stopping(val_loss, epoch)
        if abs(val_loss - early_stopping.best_loss) < 1e-6:
            utils.save_model(P, d, args, train_acc, val_acc, val_loss, epoch, delete=True, add="bm")


        """ Early stopping. """
        if early_stopping.early_stop:
            utils.lp("Early stopping!", P)
            break
    
    print("\nTraining done!")
    print("Testing model...")
    # Test model
    model.eval()
    with torch.no_grad():
        test_loss, test_acc = trainutils.test_model(early_stopping.best_epoch, P, d)
    utils.lp(f"LOSS test: {test_loss:.4f}", P)
    utils.lp(f" ACC test: {test_acc:.4f}", P)

    utils.update_model_with_test_results(P, d, test_acc, test_loss)

    


if __name__ == "__main__":

    """ Import parameters from config."""
    # Constants
    P = V1_5_train_cfg.PARAMS
    # Dynamic Parameters
    d = {}

    start_time = time.time()

    # Set seed if specified.
    utils.set_seed(P["SEED"]) if P["SEED"] else None

    # Main
    main(P, d, start_time)

    # Close Tensorboard writer to flush any remaining logs.
    d["tb_writer"].close()

    # Print time taken.
    time_taken = utils.get_time_elapsed(start_time)
    utils.lp(f"\nTraining took {time_taken}.", P)