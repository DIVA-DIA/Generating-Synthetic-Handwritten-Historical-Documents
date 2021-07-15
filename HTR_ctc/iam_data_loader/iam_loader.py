'''
@author: georgeretsi
'''
import math
import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from skimage import io as img_io
from matplotlib import cm
from PIL import Image, ImageOps
from torch.autograd import Variable
from os.path import isfile
import torchvision.transforms as transforms

try:
    from utils.auxilary_functions import image_resize, centered
except:
    from HTR_ctc.utils.auxilary_functions import image_resize, centered

from .iam_config import *
from .iam_utils import gather_iam_info
def main_loader(set, level):

    info = gather_iam_info(set, level)

    data = []
    print(len(info))
    for i, (img_path, transcr) in enumerate(info):

        if i % 1000 == 0:
            print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))

        try:
            #img2 = img_io.imread(img_path + '.png')
            #img2 = 1 - img2.astype(np.float32) / 255.0
            img = cv2.normalize(cv2.imread(img_path + '.png'), None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            data += [(img.copy(), transcr.replace("|", " "))]
        except:
            continue

        #if len(transcr) > 5:


    if set == 'train':
        data = data[:int(0.8*len(data))]
    else:
        data = data[int(0.8 * len(data)):]
    print(len(data))
    return data

class IAMLoader(Dataset):

    def __init__(self, set, level='word', fixed_size=(128, None)):

        self.fixed_size = fixed_size
        self.set = set
        save_file = dataset_path + '/' + set + '_' + level + '_RGB.pt'

        if isfile(save_file) is False:
            # hardcoded path and extension

            # if not os.path.isdir(img_save_path):
            #    os.mkdir(img_save_path)

            data = main_loader(set=set, level=level)
            torch.save(data, save_file)
        else:
            data = torch.load(save_file)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0]
        transcr = self.data[index][1]


        # add augmentation
        # ?? elastic, morphology, resizing
        if(self.set == 'test2'):
            nheight = self.fixed_size[0]
            nwidth = self.fixed_size[1]
            if nheight is None:
                nheight = img.shape[0]
            if nwidth is None:
                nwidth = int(img.shape[1] * nheight / img.shape[0])

            img = image_resize(img, height=nheight-16, width=nwidth)
           # img = centered(img, (nheight, int(1.2 * nwidth) + 32))
            #img = torch.Tensor(img).float()

        return img, transcr

class alignCollate(object):

    def __init__(self, imgH=32, imgW=None, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:

                image_shape = image.shape
                ratios.append(image_shape[1] / float(image_shape[0]))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = max(int(np.floor(max_ratio * imgH)),4)

        transform = resizeNormalize((imgW, imgH))

        #a = transform(Image.fromarray((image * 255).astype('uint8')))
        #

        images = [transform(image) for image in images]
        #np_img = images[0].float().detach().permute(1, 2, 0).cpu().numpy() * 255
        #cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/png.png', np_img)
        images = torch.cat([t.unsqueeze(0) for t in images], 0).float()  #

        return Variable(torch.Tensor(images)), labels#Variable(torch.Tensor(new_images)), labels

class alignCollate2(object):

    def __init__(self, imgH=32, imgW=None, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        new_images = []
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for img in images:
                nheight = self.imgH
                nwidth = self.imgW
                if nheight is None:
                    nheight = img.shape[0]
                if nwidth is None:
                    nwidth = int(np.random.uniform(.8, 1.2) * img.shape[1] * nheight / img.shape[0])

                img = image_resize(img, height=nheight - 16, width=nwidth)
                cv2.imwrite('/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/data/test.png',img)
                img = img[ np.newaxis, ...]
                # img = centered(img, (nheight, int(1.2 * nwidth) + 32))
                #img = Variable(torch.Tensor(img).float().unsqueeze(0))
                new_images.append(img)

        return Variable(torch.Tensor(new_images)), labels


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # Resize image as necessary to new height, maintaining aspect ratio
        o_size = img.shape
        AR = o_size[1] / float(o_size[0])
        img = cv2.resize(img, (int(math.floor(AR * self.size[1])), self.size[1]))
        o_size = img.shape
        #cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/png.png', img*255)
        #img = img.resize((int(round(AR * self.size[1])), self.size[1]), self.interpolation)
        if self.size[0] != o_size[1]:
            ones = np.ones((self.size[1], self.size[0] - o_size[1]), dtype=img.dtype)
            img = np.concatenate((img, ones), axis=1)


        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)

        return img