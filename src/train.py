import os
import argparse
import time

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

from config import Config
from model.custom_model import CustomModel
from train_tools.helper_funcs import (
    init_logger,
    get_test_file_path,
    get_train_file_path,
    get_transforms,
    valid_fn,
    timeSince,
    get_score,
    AverageMeter
)
from train_tools.dataset import TrainDataset


def train_loop(folds, fold, logger, device, output):

    logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[Config.target_col].values

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data="train"))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data="train"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if Config.scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=Config.factor,
                patience=Config.patience,
                verbose=True,
                eps=Config.eps,
            )
        elif Config.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=Config.T_max, eta_min=Config.min_lr, last_epoch=-1
            )
        elif Config.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=Config.T_0,
                T_mult=1,
                eta_min=Config.min_lr,
                last_epoch=-1,
            )
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(Config, pretrained=True)
    model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=Config.lr,
        weight_decay=Config.weight_decay,
        amsgrad=False,
    )
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.0
    best_loss = np.inf

    for epoch in range(Config.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, scheduler, device
        )

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        logger.info(f"Epoch {epoch+1} - Score: {score:.4f}")

        if score > best_score:
            best_score = score
            logger.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                os.path.join(output, f"{Config.model_name}_fold{fold}_best_score.pth"),
            )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            logger.info(f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                os.path.join(output, f"{Config.model_name}_fold{fold}_best_loss.pth"),
            )

    valid_folds["preds"] = torch.load(
        os.path.join(output, f"{Config.model_name}_fold{fold}_best_score.pth"),
        map_location=torch.device("cpu"),
    )["preds"]

    return valid_folds


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    if Config.apex:
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        if Config.apex:
            with autocast():
                y_preds = model(images)
                loss = criterion(y_preds.view(-1), labels)
        else:
            y_preds = model(images)
            loss = criterion(y_preds.view(-1), labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        if Config.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), Config.max_grad_norm
        )
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            if Config.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % Config.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                #'LR: {lr:.6f}  '
                .format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    grad_norm=grad_norm,
                    # lr=scheduler.get_lr()[0],
                )
            )
    return losses.avg


def main(train_df: pd.DataFrame, output: str, device, logger):

    """
    Prepare: 1.train
    """

    def get_result(result_df, logger):
        preds = result_df["preds"].values
        labels = result_df[Config.target_col].values
        score = get_score(labels, preds)
        logger.info(f"Score: {score:<.4f}")

    if Config.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(Config.n_fold):
            if fold in Config.trn_fold:
                _oof_df = train_loop(train_df, fold, logger, device, output)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        logger.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(os.path.join(output, "oof_df.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training G2Net model")
    parser.add_argument("--output", type=str, help="output directory")
    args = parser.parse_args()

    train_df = pd.read_csv("dataset/training_labels.csv")
    test_df = pd.read_csv("dataset/sample_submission.csv")

    train_df["file_path"] = train_df["id"].apply(get_train_file_path)
    test_df["file_path"] = test_df["id"].apply(get_test_file_path)

    # -----------------------------------------------------------------------------
    # CV Split
    # -----------------------------------------------------------------------------
    Fold = StratifiedKFold(
        n_splits=Config.n_fold, shuffle=True, random_state=Config.seed
    )
    for n, (train_index, val_index) in enumerate(
        Fold.split(train_df, train_df[Config.target_col])
    ):
        train_df.loc[val_index, "fold"] = int(n)

    train_df["fold"] = train_df["fold"].astype(int)

    # -----------------------------------------------------------------------------
    # setting
    # -----------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = init_logger(log_file=os.path.join(args.output, "train.log"))

    # -----------------------------------------------------------------------------
    # main loop
    # -----------------------------------------------------------------------------
    main(train_df=train_df, output=args.output, device=device, logger=logger)
