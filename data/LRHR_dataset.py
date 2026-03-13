from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import cv2
from torchvision.transforms import functional as trans_fn
from torchvision.transforms import InterpolationMode
import re


class LRHRDataset(Dataset):
    def __init__(self, dataroot, img_high, img_width, split='train', data_len=-1, need_LR=False, style_ref_mode='paired'):
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.img_high = img_high
        self.img_width = img_width
        self.style_ref_mode = style_ref_mode

        self.sr_path = Util.get_paths_from_images('{}/abnormal'.format(dataroot))
        self.hr_path = Util.get_paths_from_images('{}/normal'.format(dataroot))
        self.dataset_len = len(self.hr_path)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

        if self.style_ref_mode not in ['paired', 'random']:
            raise ValueError('style_ref_mode must be one of [paired, random], got {}'.format(self.style_ref_mode))

    def __len__(self):
        return self.data_len

    def _read_and_resize(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img)
        img = trans_fn.resize(img, (self.img_width, self.img_high), interpolation=InterpolationMode.BICUBIC)
        return img

    def _sample_style_index(self, index):
        if self.style_ref_mode == 'paired' or self.dataset_len <= 1:
            return index

        style_index = random.randint(0, self.dataset_len - 1)
        if self.dataset_len > 1:
            while style_index == index:
                style_index = random.randint(0, self.dataset_len - 1)
        return style_index

    def __getitem__(self, index):
        string = str(self.sr_path[index])
        pattern = r"image-(\d{4})"
        hr_number_d4 = re.findall(pattern, string)
        number = int(hr_number_d4[0]) if len(hr_number_d4) > 0 else index

        style_index = self._sample_style_index(index)

        img_hr = self._read_and_resize(self.hr_path[index])
        img_sr = self._read_and_resize(self.sr_path[index])
        img_style = self._read_and_resize(self.hr_path[style_index])

        img_sr, img_hr, img_style = Util.transform_augment(
            [img_sr, img_hr, img_style], split=self.split, min_max=(-1, 1)
        )

        return {
            'HR': img_hr,
            'SR': img_sr,
            'STYLE_REF': img_style,
            'Index': index,
            'number': number,
        }
