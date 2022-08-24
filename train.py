import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from loss import create_criterion
import madgrad
from tqdm import tqdm
import wandb

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(args):
    seed_everything(args.seed) #def: 42

    if os.path.exists(args.save_dir) is not True:
        os.mkdir(args.save_dir)
    if os.path.exists(args.model_dir) is not True:
        os.mkdir(args.model_dir)

    if args.dataset == "ten_class_dataset":
        data_dir = "/mnt/ramdisk/ten_class_dataset/train_val"
    elif args.dataset == "ten_class_dataset_half_1":
        data_dir = "/mnt/ramdisk/ten_class_dataset/train_val_half_1"
    elif args.dataset == "ten_class_dataset_half_2":
        data_dir = "/mnt/ramdisk/ten_class_dataset/train_val_half_2"
    args.dataset = "ten_class_dataset"

    # -- settings
    use_cuda = torch.cuda.is_available()
    print("USE CUDA> ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default:
    train_set = dataset_module(
        data_dir=os.path.join(data_dir, 'train'),
        resize=args.resize,
        val_ratio=0,
    )
    valid_set = dataset_module(
        data_dir=os.path.join(data_dir, 'val'),
        resize=args.resize,
        val_ratio=0,
    )

    #print(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        valid_set,
        batch_size=args.valid_batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: resnet50
    model = model_module(
        num_classes=train_set.num_classes,
    ).to(device)
    #print(model)
    
    if args.model_load_path is not None:
        model.load_state_dict(torch.load(args.model_load_path))
    #model = torch.nn.DataParallel(model)
    #torch.nn.DataParallel : https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy

    # -- optimizer
    try :
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    except AttributeError :
        opt_module = getattr(import_module("madgrad"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    # -- scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # -- logging
    logger = SummaryWriter(log_dir=args.save_dir)
    with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0.0
    best_val_loss = np.inf
    best_val_f1 = 0.0

    wandb.init(project="test-project", entity="repara_tmp")
    wandb.config = {
            "seed": args.seed,
            "learning_rate": args.lr,
            "dataset": args.dataset,
            "resize": args.resize,
            "batch_size": args.batch_size,
            "valid_batch_size": args.valid_batch_size,
            "model": args.model,
            "optimizer": args.optimizer,
            "lr": args.lr,
            #"val_ratio": args.val_ratio,
            "loss": args.criterion,
            #"lr_decay_step": args.lr_decay_step,
            #"gamma": args.gamma
    }

    # -- train starts
    for epoch in tqdm(range(args.epochs)):
        # -- train loop
        print()
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in tqdm(enumerate(train_loader)):
            inputs, labels = train_batch
            #print(inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()

            labels = labels.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            train_f1 = f1_score(labels, preds, average='macro')

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.4%} || lr {current_lr} || "
                    f"F1_score {train_f1:4.4} "
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1_score", train_f1, epoch * len(train_loader) + idx)
                wandb.log({
                    "Train/loss": train_loss,
                    "Train/accuracy": train_acc,
                    "Train/f1_score": train_f1
                })
                loss_value = 0
                matches = 0

        # -- val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []

            for val_batch in tqdm(val_loader):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()

                labels = labels.cpu().detach().numpy()
                preds = preds.cpu().detach().numpy()
                f1_item = f1_score(labels, preds, average='macro')

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                val_f1_items.append(f1_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 = np.sum(val_f1_items) / len(val_loader)

            wandb.log({
                "Val/loss": val_loss,
                "Val/acc": val_acc,
                "Val/f1": val_f1,
            })

            best_val_acc = max(best_val_acc, val_acc)

            if val_loss < best_val_loss:
                print(f"New best model for val_loss : {val_loss:4.4}! saving the best loss model..")
                torch.save(model.state_dict(), f"{args.model_dir}/{args.model}_epoch{epoch+1}_loss_{val_loss}.pth")
                best_val_loss = val_loss
            if val_f1 > best_val_f1:
                print(f"New best model for val_F1_score : {val_f1:4.4}! saving the best F1_score model..")
                torch.save(model.state_dict(), f"{args.model_dir}/{args.model}_epoch{epoch+1}_f1_{val_f1}.pth")
                
                best_val_f1 = val_f1

            scheduler.step(val_loss)  # lr scheduler

            print(f"[Val] loss: {val_loss:4.4}, F1_score {val_f1:4.4}, acc : {val_acc:4.4}% || "f"best loss: {best_val_loss:4.4}, best_F1_score {best_val_f1:4.4} , best acc : {best_val_acc:4.4}% ")
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1_score", val_f1, epoch)
            print()

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--dataset', required=True, type=str, help='dataset type (ten_class_dataset)')
    parser.add_argument('--resize', type=tuple_type, default="(224,224)", help='image resize values, (default:"(224, 224)")')
    parser.add_argument('--batch_size', type=int, default=12, help='input batch size for training (default: 12)')
    parser.add_argument('--valid_batch_size', type=int, default=12, help='input batch size for validing (default: 12)')
    parser.add_argument('--model', type=str, default='resnet101', help='model type (default: resnet101)')
    parser.add_argument('--model_load_path', type=str, default=None, help='pretrained model path (default: None)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--log_interval', type=int, default=16, help='how many batches to wait before logging training status')
    # parser.add_argument('--lr_decay_step', type=int, default=2, help='learning rate scheduler(stepLR) deacy step (default: 2)')
    # parser.add_argument('--gamma', type=float, default=0.75, help='learning rate scheduler(stepLR) gamma (default: 0.75)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)') #valid set has been hold

    # Container environment
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '../trained_models'))
    parser.add_argument('--save_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '../save_dir'))
    args = parser.parse_args()

    #print(args)
    train(args)
