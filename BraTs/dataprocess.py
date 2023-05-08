import numpy as np 
import SimpleITK as sitk
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2 
from tqdm import tqdm

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

"""
这个normalize函数也可以用于标准化图像，但它的方法略有不同。
这个函数首先通过计算给定百分位数的阈值来裁剪图像的像素值，然后针对非零像素值计算均值和标准差，对图像进行Z-score标准化。

这个函数的特点如下：

使用np.percentile函数确定上下阈值，裁剪像素值。默认情况下，它使用99%分位数作为上限，1%分位数作为下限。
在执行标准化之前，使用np.clip函数将图像的像素值限制在上述计算的上下阈值之间。
计算非零像素值的均值和标准差，然后使用Z-score标准化方法对图像进行标准化。
将标准化后的最小值替换为-9，以便在后续处理中跟踪和丢弃原始图像中的零强度像素。
这个normalize函数有一个特定的应用场景，即处理具有非零像素值的图像区域，这在医学图像处理中很常见。
因此，这个函数可以用作图像标准化，但请注意，它与之前提到的normalize_image函数的目的和方法略有不同。

"""

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9
        return tmp
    
def resize_image(image, new_shape):
    """
    Resize 3d image to new_shape in dim 1 and 2
    e.g. (160, 192, 192) -> (160, 160, 160) 
    """
    old_shape = image.shape
    resized_image = np.zeros(new_shape)

    # loop over slices and resize each slice individually
    for i in range(old_shape[0]):
        resized_image[i] = cv2.resize(image[i], (new_shape[1], new_shape[2]), interpolation=cv2.INTER_LINEAR)

    return resized_image

def read_brats_images(sub_path):
    """
    Read the brats images
    BraTs dataset has four modalities: t1, t1ce, t2, flair and one mask image in a subfolder
    Read the images and return a dictionary form like: {'modality': numpy array}
    """
    # read the subfiles of sub_path, usually it has 5 files: flair, t1, t1ce, t2 and mask
    sub_files = file_name_path(sub_path, dir=False, file=True)
    arrays = {}
    for sub_file in sub_files:
        if 'flair' in sub_file:
            flair = sitk.ReadImage(os.path.join(sub_path, sub_file))
            flair_array = sitk.GetArrayFromImage(flair)
            arrays['flair'] = flair_array
        elif 't1.nii' in sub_file:
            t1 = sitk.ReadImage(os.path.join(sub_path, sub_file))
            t1_array = sitk.GetArrayFromImage(t1)
            arrays['t1'] = t1_array
        elif 't1ce' in sub_file:
            t1ce = sitk.ReadImage(os.path.join(sub_path, sub_file))
            t1ce_array = sitk.GetArrayFromImage(t1ce)
            arrays['t1ce'] = t1ce_array
        elif 't2' in sub_file:
            t2 = sitk.ReadImage(os.path.join(sub_path, sub_file))
            t2_array = sitk.GetArrayFromImage(t2)
            arrays['t2'] = t2_array
        elif 'seg' in sub_file:
            mask = sitk.ReadImage(os.path.join(sub_path, sub_file))
            mask_array = sitk.GetArrayFromImage(mask)
            arrays['mask'] = mask_array
    return arrays 

def divide_pathces(arrays, path_key, stride=16, patch_size=32):
    """
    divide image into patches in a slide window way at the first dimension
    e.g. A (160, 160, 160) image could be divided into at most 9 (32, 160, 160) pathces
    """
    image_dict = {key: [] for key in arrays}
    for key, value in arrays.items():
        # if is train set
        if 'train' in path_key:
            for x in range(0, value.shape[0] - patch_size, stride):
                # if all slices in the patch are 0, then skip this patch
                if np.max(arrays['mask'][x: x + patch_size, :, :]) != 0:
                    patch = value[x: x + patch_size, :, :]
                    image_dict[key].append(patch)
        # if is test set, then just split the image into five same pathces, use numpy function
        else:
            patch = np.array_split(value, 5, axis=0)
            image_dict[key].extend(patch)
    # convert into numpy array
    for key, value in image_dict.items():
        image_dict[key] = np.array(value)

    return image_dict

def process_brats_image(arrays, resize=False, divide=False, path_key=None):
    """
    concat: 
        The original image shape (155, 240, 240), insert slices to make it (160, 240, 240)
        This is because make the shape divided by 2^4 could be convenient for the U-Net
        Otherwise, you can add padding in UNet decoder part
    normalize:
        normalize each modality image
    crop:
        crop the image to (160, 180, 180) to remove the zeros in the edge
    resize (optional):
        resize it to (160, 160, 160), this is because the GPU memory is limited 
    divide into patches (optional): 
        Also because of the limited GPU memory, we divide the image into patches
        (8G GPU could only support (32, 160, 160) with batch size 1 using 3D U-Net)
    """
     
    for key, value in arrays.items():
        # concat
        value = np.concatenate((np.zeros((3, 240, 240)), value), axis=0)
        value = np.concatenate((value, np.zeros((2, 240, 240))), axis=0)
        # crop
        value = crop_center(value, 192, 192)
        # normalize
        if key != 'mask':
            value = normalize(value)
        # resize
        if resize:
            value = resize_image(value, (value.shape[0], 144, 144))
        arrays[key] = value
    # divide into patches
    if divide and path_key:
        image_dict = divide_pathces(arrays, path_key)
        return image_dict
    return arrays


if __name__ == '__main__':
    # input data paths
    input_paths = {'train_hgg': './original/MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/HGG', 
                'train_lgg': './original/MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/LGG', 
                'test_hgg': './original/MICCAI_BraTS_2019_Data_Training/HGG', 
                'test_lgg': './original/MICCAI_BraTS_2019_Data_Training/LGG'}

    # output data paths
    output_paths = {'train_image': './processed/train_image', 
                    'train_mask': './processed/train_mask', 
                    'test_image': './processed/test_image', 
                    'test_mask': './processed/test_mask'}

    # create output paths 
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # the subfiles of input data paths
    file_path_list = {  'train_hgg': file_name_path(input_paths['train_hgg']), 
                        'train_lgg': file_name_path(input_paths['train_lgg']),
                        'test_hgg': file_name_path(input_paths['test_hgg']),
                        'test_lgg': file_name_path(input_paths['test_lgg'])}
    
    for path_key, path_value in file_path_list.items():
        for i in tqdm(range(len(path_value))):
            sub_path = os.path.join(input_paths[path_key], path_value[i])
            # read the image into a dictionary
            arrays = read_brats_images(sub_path)
            # process the image
            image_dict = process_brats_image(arrays, resize=True, divide=True, path_key=path_key)

            patch_num, patch_depth, patch_width, patch_height = image_dict['flair'].shape
            # print(patch_num, patch_depth, patch_height, patch_width)
            
            # merge and save the patches, merge four modalities into one tensor
            """
            save different modalities in the same patch together, one patch is like: 
            four_images: (4, patch_depth, patch_width, patch_height), we have four different modalities in this dataset
            mask: (3, patch_depth, patch_width, patch_height), we have three different masks in this dataset
            """
            for j in range(patch_num):
                if 'train' in path_key:
                    img_path = os.path.join(output_paths['train_image'], path_value[i]+ '_' + str(j) + '.npy')
                    mask_path = os.path.join(output_paths['train_mask'], path_value[i]+ '_' + str(j) + '.npy')
                else:
                    img_path = os.path.join(output_paths['test_image'], path_value[i]+ '_' + str(j) + '.npy')
                    mask_path = os.path.join(output_paths['test_mask'], path_value[i]+ '_' + str(j) + '.npy')
                
                four_images = np.stack((image_dict['flair'][j], image_dict['t1'][j], image_dict['t1ce'][j], image_dict['t2'][j]), axis=0)
                mask = image_dict['mask'][j]
                # 将mask分为WT_mask, TC_mask, ET_mask三类
                WT_mask = np.zeros(mask.shape)
                WT_mask[mask > 0] = 1
                TC_mask = np.zeros(mask.shape)
                TC_mask[mask == 1] = 1
                TC_mask[mask == 4] = 1
                ET_mask = np.zeros(mask.shape)
                ET_mask[mask == 4] = 1
                # merge the mask
                mask = np.stack((WT_mask, TC_mask, ET_mask), axis=0)
                # save the image and mask
                np.save(img_path, four_images)
                np.save(mask_path, mask)
