import argparse
import logging
import torch.cuda

from torchvision import transforms
import os
import cv2
import numpy as np
from torch.autograd import Variable

import sys, traceback
import wandb
import glob

from numpy.ma import indices
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

try:
    from iam_data_loader.iam_loader import IAMLoader, alignCollate, alignCollate2
    from generated_data_loader.models import Generator
    from models.crnn import CRNN
    from HTR_ctc.models.htr_net import HTRNet
except:
    from HTR_ctc.iam_data_loader.iam_loader import IAMLoader, alignCollate, alignCollate2
    from HTR_ctc.generated_data_loader.models import Generator
    from HTR_ctc.models.crnn import CRNN
    from HTR_ctc.models.htr_net import HTRNet
try:
    from config import *
    from reading_discriminator import *
except:
    from HTR_ctc.train_code.config import *
    from HTR_ctc.train_code.reading_discriminator import *

#try:
#    from utils.auxilary_functions import image_resize, centered
#except:
#    from HTR_ctc.utils.auxilary_functions import image_resize, centered




logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('PHOCNet-Experiment::train')
logger.info('--- Running PHOCNet Training ---')

class SaintGallLoader(Dataset):

    def __init__(self, pages=10):

        word_location = open(
            "/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/ground_truth/word_location.txt",
            "r")
        word_transcription = open(
            "/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/ground_truth/transcription.txt",
            "r")
        data = []
        transcription = word_transcription.readline().split(' ')

        for word_info in word_location:
            word_array = word_info.split(' ')

            rightPageNames = open(
                "/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/sets/valid" + str(pages) +".txt",
                "r")
            rightPage = False
            for img_name in rightPageNames:
                if img_name.replace('\n', '') in word_array[0]:
                    rightPage = True

            if not rightPage:
                continue

            while transcription[0] != word_array[0]:
                transcription = word_transcription.readline().split(' ')

            word_text = transcription[1].split('|')
            count = 0
            image = cv2.normalize(cv2.imread(
                '/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/data/line_images_normalized/' + str(
                    word_array[0]) + '.png', cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #image = image[:, :, np.newaxis]

            for word_position in word_array[2].split('|'):
                data.append([image[:, int(word_position.split('-')[0]): int(word_position.split('-')[1])].copy(), word_text[count].replace("-", "").replace("et", "&")])

                count += 1

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0]
        transcr = self.data[index][1]

        return img, transcr



def InitStandardRD(wandb = None, nlr=1e-4, rd_low_loss_learn=False, load_model = load_model_name):
    logger.info('Loading dataset')


    # load CNN
    logger.info('Preparing Net')
    net = CRNN(1, len(classes), 256)
    if (wandb is not None):
        wandb.watch(net)
    #net = HTRNet(cnn_cfg, rnn_cfg, len(classes))#




    loss = torch.nn.CTCLoss()
    net_parameters = net.parameters()

    optimizer = torch.optim.Adam(net_parameters, nlr, weight_decay=0.00005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5 * max_epochs), int(.75 * max_epochs)])

    logger.info('Initializing Reading Discriminator')
    rd = ReadingDiscriminator(optimizer, net, loss, 1e-4, load_model_full_path=load_model, rd_low_loss_learn=rd_low_loss_learn)
    return rd, scheduler

def getCrop(image, x_axis, y_axis):
    boundingBox_size = 256

    croppedImage = image[y_axis: y_axis + boundingBox_size, x_axis: x_axis + boundingBox_size,:]


    return croppedImage


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    # - train arguments
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--data_set', '-ds', choices=['IAM', 'Generated'], default='Generated',
                        help='Which dataset is used for training')
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--display', action='store', type=int, default=50,
                        help='The number of iterations after which to display the loss values. Default: 100')
    parser.add_argument('--save_number', action='store', type=int, default=0,
                        help='Specifies the number of the run that is used, if 0 no run is taken')

    parser.add_argument('--pages', choices=['1', '2', '5', '10'], default='10',
                        help='How many pages are used for training')

    # - experiment arguments
    #parser.add_argument('--load_model', '-lm', action='store', default=None,
    #                    help='The name of the pretrained model to load. Defalt: None, i.e. random initialization')
    #parser.add_argument('--save_model', '-sm', action='store', default='whateva.pt',
    #                    help='The name of the file to save the model')
    #os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    args = parser.parse_args()
    augment_factor = 0
    rd_pretrain_load = '/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/saved_models/IAM.pt'
    pages = int(args.pages)

    wandb.init(project="saint-gall", config={"pages": pages, "rd_pretrain_load":rd_pretrain_load })

    wandb.run.name = 'Full_Pages_' + str(pages) + '_PreIAM'
    wandb.run.save()

    transforms_ = [transforms.ToTensor()]
    transform = transforms.Compose(transforms_)


    rd, _ = InitStandardRD(wandb=wandb, nlr=args.learning_rate, load_model=rd_pretrain_load)
    train_loader = DataLoader(SaintGallLoader(pages), batch_size=32, shuffle=True, num_workers=8, collate_fn=alignCollate2(imgH=48, imgW=128, keep_ratio=True))

    for epoch in range(0,81):
        print('Epoch:' + str(epoch))
        rd.train_on_Dataloader(epoch,train_loader, wandb=wandb)
        if epoch % 20 == 0:
            overall_cer_path, overall_cer_beam, overall_wer_path, overall_wer_beam, _, _ = rd.testOnSaintGall()
            wandb.log({'overall_cer_path' : overall_cer_path, 'epoch' : epoch,
                       'overall_cer_beam': overall_cer_beam,
                       'overall_wer_path':overall_wer_path,
                       'overall_wer_beam': overall_wer_beam})
            rd.saveModel('',model_path='/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/saved_models/clean_rd_' + str(pages) + '.pt')