import yaml
import os 
import shutil
import argparse
import wandb
import glob 
from tqdm import tqdm 

import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import models
import datasets
import utils
import copy
import gc 

def main(config):
    save_path = os.path.join('logs', config['dataset'] + '_' + config['model'] + '_')
    utils.is_path(save_path)
    utils.set_log_path(save_path)
    
    # train_files 
    image_files = glob.glob(os.path.join(config['dataset_dir'], 'train_image/*.npy'))
    mask_files = glob.glob(os.path.join(config['dataset_dir'], 'train_mask/*.npy'))

    print("image number: ", len(image_files))
    print("mask number: ", len(mask_files))
    # dataset 
    train_image_files, val_image_files, train_mask_files, val_mask_files = \
                train_test_split(image_files, mask_files, test_size=0.2, random_state=42)
    # print("train number: ", len(train_image_files))
    # print("val number: ", len(val_image_files))
    # print("train mask number: ", len(train_mask_files))
    # print("val mask number: ", len(val_mask_files))
    
    train_dataset = datasets.BraTsDataset(train_image_files, train_mask_files)
    val_dataset = datasets.BraTsDataset(val_image_files, val_mask_files)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    print("train image shape: ", train_dataset[0][0].shape)
    print("train mask shape: ", train_dataset[0][1].shape)
    print("type of train image: ", type(train_dataset[0][0]))
    print("type of train mask: ", type(train_dataset[0][1]))
    # model
    model = models.UNet3D(in_channels=4, out_channels=3).cuda()
    # args = None
    # model = models.UNet3D(args).cuda()

    optimizer, lr_scheduler = utils.make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])
    criterion = utils.DiceLoss()
    # train
    log_keys = ['train_loss', 'train_dice', 'val_loss', 'val_dice']

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    for epoch in range(config['epochs']):
        aves = {key: utils.AverageMeter() for key in log_keys}
        model.train()
        for i, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            dice = utils.dice_coef(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            aves['train_loss'].update(loss.item(), inputs.size(0))
            aves['train_dice'].update(dice, inputs.size(0))

            if (i % 20 == 0):
                utils.log('epoch: {}, train_loss: {:.4f}, train_dice: {:.4f}'.format(
                    epoch, aves['train_loss'].avg, aves['train_dice'].avg))
                wandb.log({'train_loss': aves['train_loss'].avg, 'train_dice': aves['train_dice'].avg})

        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                dice = utils.dice_coef(outputs, targets)

                aves['val_loss'].update(loss.item(), inputs.size(0))
                aves['val_dice'].update(dice, inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step()
        utils.log('epoch: {}, train_loss: {:.4f}, train_dice: {:.4f}, val_loss: {:.4f}, val_dice: {:.4f}'.format(
            epoch, aves['train_loss'].avg, aves['train_dice'].avg, aves['val_loss'].avg, aves['val_dice'].avg))
        wandb.log({'train_loss': aves['train_loss'].avg, 'train_dice': aves['train_dice'].avg,
                    'val_loss': aves['val_loss'].avg, 'val_dice': aves['val_dice'].avg})

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/brats_unet.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    # wandb setting
    wandb.init(project='3d_segmentation', name=config['dataset'] + '_' + config['model'])
    wandb.config.update(config)

    # path
    main(config)
