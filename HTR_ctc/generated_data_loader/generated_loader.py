from os.path import isfile

from .generated_utils import *
from .generated_config import *
import os
import traceback
import numpy as np
import cv2
from torch.utils.data import Dataset
import time
import torch
import PIL
from PIL import Image

from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian


try:
    from utils.auxilary_functions import image_resize, centered
except:
    from HTR_ctc.utils.auxilary_functions import image_resize, centered

class GeneratedLoader(Dataset):

    def __init__(self, set = 'train', augment_factor = 0, resize = False, nr_of_channels=1, fixed_size=(128, None), generator_path = '', source_dataset = 'EG-BG-LC', save_path = ''):
        self.resize = resize
        self.augment_factor = augment_factor
        self.fixed_size = fixed_size
        self.set = set
        generate_data = False
        if save_path == '':
            save_file = dataset_path + 'generated_' + set + '.pt'
        else:
            save_file = save_path + 'dataset_' + set + '.pt'

        if isfile(save_file):
            try:
                data = torch.load(save_file)
            except:
                data = []
                generate_data = True

        if isfile(save_file) is False or generate_data:
            data = []
            min_len = 1
            if set == 'train':
                min_len = 70000
            elif set == 'test':
                min_len = 2000
            while len(data) < min_len:
                try:
                    data.extend(generateCrops(nr_of_channels,'/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/data/generated/', just_generate=False, generator_path=generator_path, source_dataset= source_dataset))
                except:
                    print('Error in generation of crops!' + str(traceback.format_exc()))
                print('Data length is ' + str(len(data)))
            if generator_path != '':
                print('Save')
                #torch.save(data, os.path.dirname(generator_path) + '/dataset_' + set +'.pt')

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0].squeeze()
        transcr = self.data[index][1]


        # add augmentation
        # ?? elastic, morphology, resizing

        nheight = self.fixed_size[0]
        nwidth = self.fixed_size[1]
        if nheight is None:
            nheight = img.shape[0]

        if nwidth is None:
            nwidth = int(np.random.uniform(.8, 1.2) * img.shape[1] * nheight / img.shape[0])

        #augmentation
        noise = np.random.uniform(.1, 0.25) * self.augment_factor
        blur = np.random.uniform(.5, 2.0) * self.augment_factor

        #img = image_resize(img, height=2000, width=(int(2000/nheight)*nwidth))
        img = rotate(img, angle=np.random.random_integers(-2, 2), mode='constant', cval= 1, resize=True) # rotating

        img = random_noise(img, var=noise ** 2) # adding noise
        img = gaussian(img, sigma=blur, multichannel=True) # blurring


        #end augmentation

        # img = img.resize(int((nheight-16)/2), int(nwidth/2), PIL.Image.NEAREST)
        #
        # img = img.resize((nheight-16, nwidth), PIL.Image.NEAREST)
        if self.set == 'test':
            img = image_resize(img, height=nheight-16, width=nwidth)


        # binarization
        #img = img[:, :, np.newaxis]

        #_, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

            #img = torch.Tensor(img).float().unsqueeze(0)

        return img, transcr

