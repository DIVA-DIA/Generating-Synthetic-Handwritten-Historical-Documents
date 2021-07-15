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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=alignCollate2(imgH=48, imgW=128, keep_ratio=True))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, collate_fn=alignCollate2(imgH=48, imgW=128, keep_ratio=True))
    return train_set, test_set, train_loader, test_loader

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def InitStandardRD(wandb = None, nlr=1e-4, rd_low_loss_learn=False, load_model = load_model_name):
    logger.info('Loading dataset')


    # load CNN
    logger.info('Preparing Net')
    net = CRNN(1, len(classes), 256)
    print('RD')
    count_parameters(net)
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

    args = parser.parse_args()
    augment_factor = 0
    source_dataset = args.source_dataset
    path_to_models = '/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/output/'
    save_number = args.save_number

    if 'LC' not in source_dataset:
        forceLower = True
    else:
        forceLower = False

    if save_number == 0:
        _, test_set, train_loader, test_loader = InitDataset(args.data_set, augment_factor=augment_factor,
                                                             source_dataset=source_dataset)
        rd, scheduler = InitStandardRD(nlr=args.learning_rate)

        # img = cv2.normalize(cv2.imread('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/TestWord.png', cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        curr_max_epochs = max_epochs
        logger.info('Training:')

        for epoch in range(1, curr_max_epochs + 1):
            cer, wer = rd.train_on_Dataloader(epoch, train_loader, test_set, test_loader, scheduler,
                                              forceLower=forceLower)

            logger.info('Saving net after %d epochs', epoch)
            rd.saveModel('rd.pt')

    else:
        dir_list = glob.glob(path_to_models + str(save_number) +'/*/')
        for generator_path in dir_list:
            path_to_generator_A2B = generator_path + 'netG_A2B.pth'
            run_name = 'sg_' + str(save_number) + '_ep' + os.path.basename(os.path.normpath(generator_path) + '_aug' + str(augment_factor))
            #
            api = wandb.Api()

            htr_runs = api.runs("htr-data")

            run_exist = False
            for run in htr_runs:
                if run.name == run_name:
                    run_exist = True

            if run_exist:
                print('Epoch ' + os.path.basename(os.path.normpath(generator_path)) + ' already exists')
                continue
            # W & B init
            wandb.init(project="htr-data", reinit=True, config={"learning_rate": args.learning_rate,
                                                                "augment_factor": augment_factor,
                                                                "data_set": args.data_set,
                                                                "save_model_name": save_model_name,
                                                                "load_model_name": load_model_name,
                                                                "path_to_generator_A2B": path_to_generator_A2B,
                                                                "source_dataset": source_dataset,
                                                                "used_save_number": save_number})


            wandb.run.name = run_name
            wandb.run.save()

            # print out the used arguments
            logger.info('###########################################')
            logger.info('Experiment Parameters:')
            for key, value in vars(args).items():
                logger.info('%s: %s', str(key), str(value))
            logger.info('###########################################')

            _, test_set, train_loader, test_loader = InitDataset(args.data_set, augment_factor=augment_factor, path_to_generator_A2B=path_to_generator_A2B, source_dataset = source_dataset, save_path=generator_path)
            rd, scheduler = InitStandardRD(nlr=args.learning_rate)

            # img = cv2.normalize(cv2.imread('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/TestWord.png', cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            curr_max_epochs = max_epochs
            logger.info('Training:')
            old_cer = 1
            old_wer = 1
            try:
                for epoch in range(1, curr_max_epochs + 1):

                    cer, wer = rd.train_on_Dataloader(epoch, train_loader, test_set, test_loader, scheduler, wandb, forceLower = forceLower)

                    # never stop run on improvement
                    if epoch == curr_max_epochs + 1:
                        if cer < old_cer:
                            curr_max_epochs += 1
                    old_cer = cer
                    old_wer = wer
                    logger.info('Saving net after %d epochs', epoch)
                    rd.saveModel('rd.pt', model_path=generator_path)

            except:
                wandb.run.name = 'failed_' + run_name
                wandb.run.save()
                print(traceback.format_exception(*sys.exc_info()))
            wandb.join()




