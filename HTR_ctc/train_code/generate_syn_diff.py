import argparse
import logging
import torch.cuda

from torchvision import transforms
import os
import cv2
import numpy as np
from torch.autograd import Variable

import sys, traceback
#import wandb
import glob

from numpy.ma import indices
from torch.utils.data import DataLoader
try:
    #from iam_data_loader.iam_loader import IAMLoader, alignCollate, alignCollate2
    from generated_data_loader.models import Generator
    from models.crnn import CRNN
    from HTR_ctc.models.htr_net import HTRNet
except:
    #from HTR_ctc.iam_data_loader.iam_loader import IAMLoader, alignCollate, alignCollate2
    from HTR_ctc.generated_data_loader.models import Generator
    from HTR_ctc.models.crnn import CRNN
    from HTR_ctc.models.htr_net import HTRNet
try:
    from config import *
    #from reading_discriminator import *
except:
    from HTR_ctc.train_code.config import *
    #from HTR_ctc.train_code.reading_discriminator import *

#try:
#    from utils.auxilary_functions import image_resize, centered
#except:
#    from HTR_ctc.utils.auxilary_functions import image_resize, centered




logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('PHOCNet-Experiment::train')
logger.info('--- Running PHOCNet Training ---')

#cudnn.benchmark = True
# prepare datset loader
def InitDataset(data_set, augment_factor = 0, path_to_generator_A2B = '', source_dataset='EG-BG-LC', save_path = ''):
    if data_set == 'IAM':
        train_set = IAMLoader('train', level=data_name, fixed_size=(128, None))
        test_set = IAMLoader('test', level=data_name, fixed_size=(48, 128))
    elif data_set == 'Generated':
        train_set = GeneratedLoader(set='train', nr_of_channels=1, fixed_size=(128, None), augment_factor = augment_factor, generator_path=path_to_generator_A2B, source_dataset=source_dataset, save_path=save_path)
        test_set = GeneratedLoader(set='test', nr_of_channels=1, fixed_size=(48, 128), augment_factor = 0, generator_path=path_to_generator_A2B, source_dataset=source_dataset, save_path=save_path)
    # augmentation using data sampler
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=alignCollate2(imgH=48, imgW=128, keep_ratio=True))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, collate_fn=alignCollate2(imgH=48, imgW=128, keep_ratio=True))
    return train_set, test_set, train_loader, test_loader


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
    rd = ReadingDiscriminator(optimizer, net, loss, 1e-4, load_model, rd_low_loss_learn)
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

    parser.add_argument('--source_dataset', choices=['EG-old', 'EG-BG-LC-old', 'EG', 'EG-LC', 'EG-BG', 'EG-BG-LC'], default='EG-BG-LC',
                        help='Which source dataset to pick')

    # - experiment arguments
    #parser.add_argument('--load_model', '-lm', action='store', default=None,
    #                    help='The name of the pretrained model to load. Defalt: None, i.e. random initialization')
    #parser.add_argument('--save_model', '-sm', action='store', default='whateva.pt',
    #                    help='The name of the file to save the model')
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    args = parser.parse_args()
    augment_factor = 0
    source_dataset = 'EG-BG-LC'
    path_to_model_IAM = '/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/saved_models/IAMN90.pth'
    path_to_model_GEN = '/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/best/GEN150.pth'
    path_to_model_NoPre = '/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/best/NoPreN200.pth'
    path_to_model_IAM2RD = '/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/best/TwoIAMN110.pth'
    path_to_model_GEN2RD = '/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/best/TwoGEN190.pth'
    path_to_model_NoPre2RD = '/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/best/TwoNoPreN190.pth'
    save_number = args.save_number

    transforms_ = [transforms.ToTensor()]
    transform = transforms.Compose(transforms_)

    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(1, 3, 256, 256)
    IAM_A2B = Generator(3, 3)
    paras = sum(p.numel() for p in IAM_A2B.parameters() if p.requires_grad) / 10 ** 6

    IAM_A2B.cuda()
    IAM_A2B.load_state_dict(torch.load(path_to_model_IAM))

    # GEN_A2B = Generator(3, 3)
    # GEN_A2B.cuda()
    # GEN_A2B.load_state_dict(torch.load(path_to_model_GEN))
    #
    # NoPre_A2B = Generator(3, 3)
    # NoPre_A2B.cuda()
    # NoPre_A2B.load_state_dict(torch.load(path_to_model_NoPre))


    #IAM2RD_A2B = Generator(3, 3)
    #IAM2RD_A2B.cuda()
    #IAM2RD_A2B.load_state_dict(torch.load(path_to_model_IAM2RD))
    #
    # GEN2RD_A2B = Generator(3, 3)
    # GEN2RD_A2B.cuda()
    # GEN2RD_A2B.load_state_dict(torch.load(path_to_model_GEN2RD))
    #
    # NoPre2RD_A2B = Generator(3, 3)
    # NoPre2RD_A2B.cuda()
    # NoPre2RD_A2B.load_state_dict(torch.load(path_to_model_NoPre2RD))
    count = 0
    all_image_names = os.listdir('/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/data/generated/' + source_dataset)
    for image_name in all_image_names:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            print(image_name)
            image = cv2.imread(os.path.join('/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/data/generated/' + source_dataset, image_name))
            #print(image.size)
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            max_height, max_width, _ = image.shape
            height = 0
            width = 0
            width_list = []
            height_list = []
            counter = 0
            stepsize = 10
            while height < max_height-256:

                while width < max_width-256:
                    crop_image = getCrop(image, width, height)
                    real_A = Variable(input_A.copy_(transform(crop_image)))
                    IAM_image = (IAM_A2B(real_A).data)
                    np_IAM = cv2.normalize(IAM_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
                    if width_list == []:
                        width_list = np_IAM
                    else:
                        #a = np.concatenate((width_list, np.ones(shape=(256,stepsize,3))*255), axis=1)
                        #b = np.concatenate((np.ones(shape=(width_list.shape[0],width_list.shape[1]-(256-stepsize), 3))*255, np_IAM), axis=1);
                        x = width_list[:,0:width_list.shape[1] - stepsize,  :]
                        y_0 = width_list[:,width_list.shape[1] - stepsize:width_list.shape[1], :]
                        y_1 = np_IAM[:,0:np_IAM.shape[1] - (256-stepsize), :]
                        z = np_IAM[:,np_IAM.shape[1] - (256-stepsize):np_IAM.shape[1], :]
                        y = (y_0 + y_1)/2
                        width_list =  np.concatenate((x,y,z),axis=1)

                    width += (256-stepsize)
                width = 0
                width_array = width_list# np.concatenate(width_list, axis=1)
                if height_list == []:
                    height_list = width_list
                else:
                    #a = np.concatenate((height_list, np.ones(shape=(stepsize, height_list.shape[1], 3))*255), axis=0)
                    #b = np.concatenate((np.ones(shape=(height_list.shape[0] -(256-stepsize), height_list.shape[1], 3))*255, width_list),axis=0)

                    x = height_list[0:height_list.shape[0]-stepsize, :,:]
                    y_0 = height_list[height_list.shape[0]-stepsize:height_list.shape[0], :,:]
                    y_1 = width_list[0:width_list.shape[0] - (256-stepsize), :, :]
                    z = width_list[width_list.shape[0]-(256-stepsize):width_list.shape[0], :,:]
                    y = (y_0 + y_1)/2
                    height_list = np.concatenate((x,y,z),axis=0)
                height += (256-stepsize)
                width_list = []
                #cv2.imwrite('/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/data/output/iam' + str(counter) + '.png', height_list)
                counter += 1;

            iam_set = height_list # np.concatenate(height_list, axis = 0)
            cv2.imwrite('/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/data/generated/syn2/' + image_name, iam_set)


            #
            # image, csvCropString = getRandomCrop(image, image_name, '/home/manuel/CycleGANRD/HTR_ctc/data/generated/', source_dataset)
            # real_A = Variable(input_A.copy_(transform(image)))
            # IAM_image = (IAM_A2B(real_A).data)
            # np_IAM = cv2.normalize(IAM_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
            # GEN_image = (GEN_A2B(real_A).data)
            # np_GEN = cv2.normalize(GEN_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)* 255
            # NoPre_image = (NoPre_A2B(real_A).data)
            # np_NoPre = cv2.normalize(NoPre_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
            #
            # IAM2RD_image = (IAM2RD_A2B(real_A).data)
            # np_IAM2RD = cv2.normalize(IAM2RD_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), None, alpha=0, beta=1,
            #                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
            # GEN2RD_image = (GEN2RD_A2B(real_A).data)
            # np_GEN2RD = cv2.normalize(GEN2RD_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), None, alpha=0, beta=1,
            #                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
            # NoPre2RD_image = (NoPre2RD_A2B(real_A).data)
            # np_NoPre2RD = cv2.normalize(NoPre2RD_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), None, alpha=0,
            #                          beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255

            #iam_set = np.concatenate((image *255, np_IAM, np_IAM2RD), axis=1)
            #gen_set = np.concatenate((image *255, np_GEN,  np_GEN2RD), axis=1)
            #nopre_set = np.concatenate((image *255, np_NoPre, np_NoPre2RD), axis=1)

            #a = np.concatenate(np.array(np_IAM)[indices.astype(int)], np.array(np_IAM)[indices.astype(int)], np.array(np_IAM)[indices.astype(int)],np_GEN, np_NoPre)
            #cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/best/images2RD/iam' + str(count) + '.png', iam_set)
            #cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/best/images2RD/gen' + str(count) + '.png', gen_set)
            #cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/best/images2RD/nopre' + str(count) + '.png', nopre_set)
            #count = count +1


