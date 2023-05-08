import torch 
import torch.nn.functional as F
import os 
import shutil


log_path = None

def set_log_path(path):
    global log_path
    log_path = path

def is_path(path, remove=True):
    """
    if path exists, choose to remove it or not
    else create the path
    """
    if os.path.exists(path):
        if remove and input('{} exists, remove? ([y]/n): '.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def log(obj, filename='log.txt'):
    print(obj)
    if log_path is not None:
        with open(os.path.join(log_path, filename), 'a') as f:
            print(obj, file=f)

def compute_params(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num >= 1e6:
        return '{:.1f}M'.format(num / 1e6)
    else:
        return '{:.1f}K'.format(num / 1e3)
    

def file_name_path(file_dir, dir=True, file=False):
    """
    get the subfiles or subdirs
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            # print("sub_dirs: ", dirs)
            return dirs
        if len(files) and file:
            # print("sub_files: ", files)
            return files
        

def crop_center(img, cropx, cropy):
    """
    crop the image to the center
    img: input image
    cropx: crop size of x 
    cropy: crop size of y
    """
    y, x = img[0].shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty : starty + cropy, startx : startx + cropx]


    
def make_optimizer(params, name, lr, weight_decay=None, scheduler=None):
    """
    choose optimizer and scheduler
    """
    if weight_decay is None:
        weight_decay = 0.0
    if name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None:
        if scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        elif scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)

    return optimizer, scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def dice_coef(pred, target, smooth=1e-5):
#     num = pred.size(0)
#     pred = pred.view(num, -1)
#     target = target.view(num, -1)
#     intersection = (pred * target).sum(1)
#     dice = (2. * intersection + smooth) / (pred.sum(1) + target.sum(1) + smooth)
#     return (dice.sum() / num).item()

def dice_coef(pred, target, smooth=1e-5):
    # pred sigmoid
    pred = torch.sigmoid(pred)
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    return dice.item()

# def dice_coef(pred, target, smooth=1e-5):
#     # 将 tensor 移动到 CPU 设备并分离梯度图
#     pred = pred.detach().cpu()
#     target = target.detach().cpu()

#     num = pred.size(0)
#     pred = pred.view(num, -1)
#     target = target.view(num, -1)
#     intersection = (pred * target).sum(1)
#     dice = (2. * intersection + smooth) / (pred.sum(1) + target.sum(1) + smooth)
#     return dice.sum() / num


import torch

def dice_coef_3d(y_true, y_pred):
    # 计算TP
    intersection = (y_true * y_pred).sum()
    # 计算FP和FN
    fp = y_pred.sum() - intersection
    fn = y_true.sum() - intersection
    # 计算Dice系数
    dice = (2.0 * intersection) / (2.0 * intersection + fp + fn)
    return dice.item()

