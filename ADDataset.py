import os
import random

# matplotlib
import numpy
from matplotlib import image as mImage
# torch
from torch.utils.data import Dataset

# constants
import ad_net_constants


class ADDataset(Dataset):

    def __init__(self, _img_dir: str, _type: str, seed, transform=None):
        random.seed(seed)

        _img_dir = _img_dir if _img_dir.endswith('') else _img_dir[:-1]
        cover_dir = _img_dir + ad_net_constants.COVER_FOLDER
        stego_dir = _img_dir + ad_net_constants.STEGO_FOLDER
        image_names = os.listdir(cover_dir)
        random.shuffle(image_names)

        if len(os.listdir(cover_dir)) != len(os.listdir(stego_dir)):
            raise RuntimeError("Cover and stego folders has different size!")

        # ration train (8): test (1): validation (1)
        if _type == 'validation':
            self.image_list = image_names[:900]
        elif _type == 'test':
            self.image_list = image_names[900:1800]
        elif _type == 'train':
            self.image_list = image_names[1800:1816]
        else:
            raise RuntimeError('Only `validation, test, train` values available!')

        self.type = _type
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.transform = transform
        self.self_name = f'\'{self.type.capitalize()} dataset\''

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        name = self.image_list[index]

        cover_path = os.path.join(self.cover_dir, name)
        stego_path = os.path.join(self.stego_dir, name)

        cover_img = mImage.imread(cover_path, 'jpeg')
        stego_img = mImage.imread(stego_path, 'jpeg')

        # normalizing the pixel values
        cover_data = numpy.asarray(cover_img, dtype='float32')
        stego_data = numpy.asarray(stego_img, dtype='float32')

        entry = {
            'images': numpy.stack([cover_data, stego_data]),
            'label': numpy.array([ad_net_constants.COVER_SIGNAL, ad_net_constants.STEGO_SIGNAL], dtype='int32')
        }

        if self.transform:
            entry = self.transform(entry)

        return entry

    def info(self) -> str:
        return f'Data set {self.self_name} info:\n\t' \
               f'type: {self.type}\n\t' \
               f'size: {self.__len__()}' \
               f'\n\timages: {self.image_list}\n'
