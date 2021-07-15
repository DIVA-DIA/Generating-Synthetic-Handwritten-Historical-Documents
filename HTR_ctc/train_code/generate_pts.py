import argparse
import logging
import torch.cuda
import sys, traceback
import wandb
import glob
from torch.utils.data import DataLoader
try:
    from iam_data_loader.iam_loader import IAMLoader, alignCollate, alignCollate2
    from generated_data_loader.generated_loader import GeneratedLoader
    from models.crnn import CRNN
    from HTR_ctc.models.htr_net import HTRNet
except:
    from HTR_ctc.iam_data_loader.iam_loader import IAMLoader, alignCollate, alignCollate2
    from HTR_ctc.generated_data_loader.generated_loader import GeneratedLoader
    from HTR_ctc.models.crnn import CRNN
    from HTR_ctc.models.htr_net import HTRNet
try:
    from config import *
    from reading_discriminator import *
except:
    from HTR_ctc.train_code.config import *
    from HTR_ctc.train_code.reading_discriminator import *

try:
    from utils.auxilary_functions import image_resize, centered
except:
    from HTR_ctc.utils.auxilary_functions import image_resize, centered



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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=alignCollate(imgH=48, imgW=128, keep_ratio=True))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, collate_fn=alignCollate(imgH=48, imgW=128, keep_ratio=True))
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


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    # - train arguments
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--data_set', '-ds', choices=['IAM', 'Generated'], default='Generated',
                        help='Which dataset is used for training')
    parser.add_argument('--rd_pretrain_load', choices=[None, 'IAM.pt', 'GEN_EG_BG_LC.pt', 'GEN_EG_BG.pt'], default=None,
                        help='Which pretraining for reading discriminator to use')
    parser.add_argument('--source_dataset', choices=['EG', 'EG-LC', 'EG-BG', 'EG-BG-LC'], default='EG-BG-LC',
                        help='Which source dataset to pick')

    args = parser.parse_args()
    augment_factor = 0
    source_dataset = args.source_dataset
    path_to_models = '/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/output/'

    if source_dataset == 'EG' or source_dataset == 'EG-BG':
        forceLower = True
    else:
        forceLower = False

    rd, scheduler = InitStandardRD(nlr=args.learning_rate, load_model=args.rd_pretrain_load)
    rd.testOnHistorical(forceLower)





