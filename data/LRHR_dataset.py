from io import BytesIO
import sre_parse
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import cv2
from torchvision.transforms import functional as trans_fn
from torchvision.transforms import InterpolationMode
import re,torch
import torch.nn.functional as F
from PIL import Image, ImageOps




# def resize_and_convert(img, size, resample):
#     if(img.size[0] != size):
#         img = trans_fn.resize(img, size, resample) #首先将图像调整为目标尺寸，使用指定的重采样方法
#         img = trans_fn.center_crop(img, size)  # 将图像进行居中裁剪，使其尺寸与目标尺寸完全匹配
#     return img 

# def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC):
#     lr_img = resize_and_convert(img, sizes[0], resample) #把img转化成size
#     hr_img = resize_and_convert(img, sizes[1], resample) # 把图片
#     sr_img = resize_and_convert(lr_img, sizes[1], resample)

    # return [lr_img, hr_img, sr_img]

def pad_to_nearest_power_of_2(x, base=32):
    w, h = x.size  
    new_w = (w + base - 1) // base * base
    new_h = (h + base - 1) // base * base
    pad_w = new_w - w
    pad_h = new_h - h
    img_padded = ImageOps.expand(x, (0, 0, pad_w, pad_h), fill=0)  # 纯黑填充
    return img_padded,(pad_h, pad_w)

class LRHRDataset(Dataset):
    def __init__(self, dataroot, img_high ,img_width, split='train' , data_len=-1, need_LR=False):
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.img_high = img_high
        self.img_width = img_width

        self.sr_path = Util.get_paths_from_images(
            '{}/abnormal'.format(dataroot)) #读取异常图片
        self.hr_path = Util.get_paths_from_images(
            '{}/normal'.format(dataroot)) #读取异常对应的正常图片

        self.dataset_len = len(self.hr_path)
        
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len
    


    def __getitem__(self, index):
        
        string = str(self.hr_path[index])
        # pattern = r"img-(\d{4})"
        #pattern = r"(?:img-(\d{4})|slice_(\d{4,}))"
        pattern = r"(?:img-|slice_)(\d{4})"
        hr_number_d4 = re.findall(pattern, string)
        # if hr_number_d4:
        #     hr_number_d4 = hr_number_d4.group(1) if hr_number_d4.group(1) else hr_number_d4.group(2)
            
        img_HR = cv2.imread(str(self.hr_path[index]), cv2.IMREAD_GRAYSCALE)
        img_HR = Image.fromarray(img_HR)
        img_HR, (pad_h, pad_w) = pad_to_nearest_power_of_2(img_HR)

        img_SR = cv2.imread(str(self.sr_path[index]), cv2.IMREAD_GRAYSCALE)
        img_SR = Image.fromarray(img_SR)
        img_SR, (pad_h, pad_w) = pad_to_nearest_power_of_2(img_SR)


        # img_HR = trans_fn.resize(img_HR, (self.img_width, self.img_high), interpolation=InterpolationMode.BICUBIC)
        # img_SR = trans_fn.resize(img_SR, (self.img_width, self.img_high), interpolation=InterpolationMode.BICUBIC)


        [img_SR, img_HR] = Util.transform_augment( 
            [img_SR, img_HR], split=self.split, min_max=(-1, 1))
        return {'HR': img_HR, 'SR': img_SR, 'Index': index, 'number': int(hr_number_d4[0]),'pad_h':pad_h,'pad_w':pad_w}
