import logging
from os.path import isfile

import numpy as np
from fast_ctc_decode import beam_search, viterbi_search
from math import log
import torch.cuda
from torch.autograd import Variable

from HTR_ctc.iam_data_loader.iam_loader import resizeNormalize
from HTR_ctc.utils.auxilary_functions import torch_augm
from HTR_ctc.train_code.config import *
import editdistance
import matplotlib.pyplot as plt
import glob
import cv2
import torchvision.transforms as transforms
from HTR_ctc.generated_data_loader.generated_utils import *
try:
    from utils.auxilary_functions import image_resize, centered
except:
    from HTR_ctc.utils.auxilary_functions import image_resize, centered


logger = logging.getLogger('Reading Discriminator')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parallelism on different graphic cards not important with batch size 1
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    net = nn.DataParallel(net)

class ReadingDiscriminator():

    def __init__(self, optimizer, net, loss, lr=1e-4, load_model = None, rd_low_loss_learn = False, load_model_full_path = None):
        self.optimizer = optimizer
        self.lr = lr
        self.net = net.to(device)
        self.loss = loss
        self.step_count = 0
        self.step_freq = 1
        if load_model is not None:
            self.loadModel(load_model)
        elif load_model_full_path is not None:
            self.loadModel('', load_model_full_path)
        self.rd_low_loss_learn = rd_low_loss_learn
        

    def train_on_Dataloader(self, epoch, train_loader, test_set = None, test_loader=None, scheduler = None, wandb = None, forceLower = False):

        self.optimizer.zero_grad()
        iter_rep = 50
        log_dict = {}
        cer_path_hist, cer_beam_hist, wer_path_hist, wer_beam_hist, cer_beam_clean_hist, wer_beam_clean_hist  = self.testOnSaintGall(forceLower)
        if test_loader is not None:
            cer_path_test, wer_path_test = self.test(epoch, test_loader)
        log_dict.update({'CERpath_hist': cer_path_hist, 'CERbeam_hist': cer_beam_hist, 'WERpath_hist': wer_path_hist, 'WERbeam_hist': wer_beam_hist,'CERbeam_clean_hist': cer_beam_clean_hist, 'WERbeam_clean_hist': wer_beam_clean_hist}) #,'CERpath_test': cer_path_test, 'WERpath_test': wer_path_test
 #

        for iter_idx, (img, transcr) in enumerate(train_loader):
            try:
                loss, _ = self.train(img, transcr)
            except:
                print('Batchsize does not fit')
            try:
                log_dict.update({'loss': loss.data})
            except:
                print('loss data not available')


            if iter_idx % iter_rep == 1 and test_set is not None:
                tst_img, tst_transcr = test_set.__getitem__(np.random.randint(test_set.__len__()))
                estimated_word = self.getResult(Variable(torch.Tensor(tst_img[np.newaxis,:,:])))
                print(tst_transcr + ' ' + estimated_word)
                if wandb is not None:
                    log_dict.update({'test_img': [wandb.Image(tst_img, caption=estimated_word + ' / ' + tst_transcr)]})
            if wandb is not None:
                wandb.log(log_dict)
            log_dict = {}

        if scheduler is not None:
            scheduler.step()

        return min(cer_path_hist, cer_beam_hist), min(wer_path_hist, wer_beam_hist)



    def train(self, img, transcr, batch_size_train=batch_size):
        # for name, param in self.net.named_parameters():
        #     print(name, param.grad)
        img = Variable(img.to(device))
        # cuda augm - alternatively for cpu use it on dataloader
        # img = torch_augm(img)
#        np_img = img.detach().squeeze(0).permute(1,2,0).cpu().numpy()
 #       cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/test2.png', np_img*255)
        output = self.net(img)


        act_lens = torch.IntTensor(batch_size_train * [output.size(0)]) #todo: batchsize * width
        try:
            labels = Variable(torch.IntTensor([cdict[c] for c in ''.join(transcr)]))
        except KeyError:
            print('Training failed because of unknown key: ' + str(KeyError))
            return -1, ''
        label_lens = torch.IntTensor([len(t) for t in transcr])

        output = output.log_softmax(2)  # .detach().requires_grad_()

        loss_val = self.loss(output.cpu(), labels, act_lens, label_lens)

        # if self.rd_low_loss_learn is true, the network only learns on a loss lower than 1
        # if off it learns on everything
        # todo: make loss_val variable or more intuitive than just 1
        self.optimizer.zero_grad()

        if not self.rd_low_loss_learn or loss_val < 1:
            loss_val.backward()
            self.optimizer.step()



            #cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/test1.png', np_img * 255)


        return loss_val, output
    
    def getResult(self, img, mode='path'):
        self.net.eval()

        img = img.unsqueeze(0) #.transpose(1, 3).transpose(2, 3)
        try:
            with torch.no_grad():
                tst_o = self.net(Variable(img.cuda()))

            if mode == 'beam':
                estimated_word, _ = beam_search(tst_o.softmax(2).cpu().numpy().squeeze(), classes, beam_size=5,
                                        beam_cut_threshold=0.012195121)
            else:
                tdec = tst_o.log_softmax(2).argmax(2).cpu().numpy().squeeze()
                t_beam = beam_search_decoder(tst_o.softmax(2).cpu().numpy().squeeze(),5)

                # todo: create a better way than to just ignore output with size [1, 1, 80] (first 1 has to be >1
                tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
                tb = [v for j, v in enumerate(t_beam) if j == 0 or v != t_beam[j - 1]]
                if tb != tt:
                    print('Unequal')

                estimated_word = ''.join([icdict[t] for t in tt]).replace('_', '').replace("v", "u")
        except:
            estimated_word = 'error'
        self.net.train()

        return estimated_word

    def test(self, epoch, test_loader):
        self.net.eval()

        logger.info('Testing at epoch %d', epoch)
        cer, wer = [], []
        for img, transcr in test_loader:
            #transcr = transcr[0]
            transcr_str = str(transcr[0]).replace("v", "u")
            img = Variable(img.to(device))
            try:
                with torch.no_grad():
                    o = self.net(img)
            except:
                print('bla')
            tdec = o.argmax(2).cpu().numpy().squeeze()
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            dec_transcr = dec_transcr.replace("v", "u")
            # tdec, _, _, tdec_len = decoder.decode(o.softmax(2).permute(1, 0, 2))
            # dec_transcr = ''.join([icdict[t.item()] for t in tdec[0, 0][:tdec_len[0, 0].item()]])

            cer += [float(editdistance.eval(dec_transcr, transcr_str)) / len(transcr_str)]
            wer += [float(editdistance.eval(dec_transcr.split(' '), transcr_str.split(' '))) / len(transcr_str.split(' '))]

        overall_cer = sum(cer) / len(cer)
        overall_wer = sum(wer) / len(wer)
        logger.info('CER at epoch %d: %f', epoch, overall_cer)
        logger.info('WER at epoch %d: %f', epoch, overall_wer)

        self.net.train()

        return overall_cer, overall_wer

    #todo: uppercase to lowercase
    def testOnHistorical(self, forceLower=False):
        wordRight = False
        files = []
        transform = transforms.Compose([transforms.ToTensor()])
        file_names = sorted(glob.glob('/HOME/pondenka/manuel/CycleGANRD/HTR_ctc/data/historical' + '/*.*'))
        cer_path, wer_path = [], []
        cer_beam, wer_beam = [], []
        cer_beam_clean, wer_beam_clean = [], []
        for name in file_names:
            img = cv2.normalize(cv2.imread(name, cv2.IMREAD_GRAYSCALE), None,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img = img[:, :, np.newaxis]
            # imgH = 48
            # h, w = img.shape
            # imgW = max(int(np.floor(w / float(h) * imgH)), 4)
            # transform2 = resizeNormalize((imgW, imgH))
            # img = transform2(img)
            # img = img[:, :, np.newaxis]
            # tens = torch.Tensor(img).transpose(1, 2).squeeze(0)


            name = name.rsplit('/')
            name = name[len(name) - 1].rsplit('.')[0]
            name = name.replace("_", "")
            name = name.replace("v", "u")
            tens = resizeImg(img, (32+16, None)).transpose(1, 3).transpose(2, 3).squeeze(0) #todo: wurde von None auf 128, gucken auf influence

            estimatedWord_path = self.getResult(tens)
            estimatedWord_beam = self.getResult(tens,'beam')


            if forceLower:
                estimatedWord_path = estimatedWord_path.lower()
                estimatedWord_beam = estimatedWord_beam.lower()

            estimatedWord_beam_clean = estimatedWord_beam.replace("-", "").replace(".", "").replace(",", "").replace(";", "").replace(":", "")

            cer_path += [float(editdistance.eval(estimatedWord_path, name)) / len(name)]
            cer_beam += [float(editdistance.eval(estimatedWord_beam, name)) / len(name)]
            cer_beam_clean += [float(editdistance.eval(estimatedWord_beam_clean, name)) / len(name)]
            #print(editdistance.eval(estimatedWord, name))
            wer_path += [float(editdistance.eval(estimatedWord_path.split(' '), name.split(' '))) / len(name.split(' '))]
            wer_beam += [float(editdistance.eval(estimatedWord_beam.split(' '), name.split(' '))) / len(name.split(' '))]
            wer_beam_clean += [float(editdistance.eval(estimatedWord_beam_clean.split(' '), name.split(' '))) / len(name.split(' '))]
            print('estimated_path: ' + estimatedWord_path + ' ; estimated_beam: ' + estimatedWord_beam + ' ; real: ' + name)
            if estimatedWord_path == name:
                wordRight = True

        overall_cer_path = sum(cer_path) / len(cer_path)
        overall_cer_beam = sum(cer_beam) / len(cer_beam)
        overall_cer_beam_clean = sum(cer_beam_clean) / len(cer_beam_clean)
        overall_wer_path = sum(wer_path) / len(wer_path)
        overall_wer_beam = sum(wer_beam) / len(wer_beam)
        overall_wer_beam_clean = sum(wer_beam_clean) / len(wer_beam_clean)

        print('Path CER: ' + str(overall_cer_path))
        print('Beam CER: ' + str(overall_cer_beam))
        print('Beam Clean CER: ' + str(overall_cer_beam_clean))
        print('Path WER: ' + str(overall_wer_path))
        print('Beam WER: ' + str(overall_wer_beam))
        print('Beam Clean WER: ' + str(overall_wer_beam_clean))
        return overall_cer_path, overall_cer_beam, overall_wer_path, overall_wer_beam, overall_cer_beam_clean, overall_wer_beam_clean


    def testOnSaintGall(self, forceLower=False):
        wordRight = False
        files = []
        transform = transforms.Compose([transforms.ToTensor()])
        cer_path, wer_path = [], []
        cer_beam, wer_beam = [], []
        cer_beam_clean, wer_beam_clean = [], []

        word_location = open("/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/ground_truth/word_location.txt", "r")
        word_transcription = open("/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/ground_truth/transcription.txt",
                                  "r")
        test_images = open("/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/sets/test.txt",
                                  "r").readline()
        data = []
        transcription = word_transcription.readline().split(' ')

        for word_info in word_location:
            word_array = word_info.split(' ')

            test_images = open(
                "/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/sets/test.txt",
                "r")
            rightPage = False
            for img_name in test_images:
                if img_name.replace('\n','') in word_array[0]:
                    rightPage = True

            if not rightPage:
                continue

            while transcription[0] != word_array[0]:
                transcription = word_transcription.readline().split(' ')

            word_text = transcription[1].split('|')
            count = 0
            image = cv2.normalize(cv2.imread('/HOME/pondenka/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/saintgalldb-v1-2.0/data/line_images_normalized/' + str(word_array[0]) + '.png', cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image = image[:, :, np.newaxis]

            for word_position in word_array[2].split('|'):
                data.append((image[:, int(word_position.split('-')[0]): int(word_position.split('-')[1]), :],
                             word_text[count]))
                count += 1


        for img, name in data:


            name = name.rsplit('/')
            name = name[len(name) - 1].rsplit('.')[0]
            name = name.replace("et", "&")
            name = name.replace("-", "")
            name = name.replace("v", "u")
            tens = resizeImg(img, (32 + 16, None)).transpose(1, 3).transpose(2, 3).squeeze(
                0)  # todo: wurde von None auf 128, gucken auf influence

            estimatedWord_path = self.getResult(tens)
            estimatedWord_beam = self.getResult(tens, 'beam')

            if forceLower:
                estimatedWord_path = estimatedWord_path.lower()
                estimatedWord_beam = estimatedWord_beam.lower()

            estimatedWord_beam_clean = estimatedWord_beam.replace("-", "").replace(".", "").replace(",",
                                                                                                    "").replace(";",
                                                                                                                "").replace(
                ":", "")

            cer_path += [float(editdistance.eval(estimatedWord_path, name)) / len(name)]
            cer_beam += [float(editdistance.eval(estimatedWord_beam, name)) / len(name)]
            cer_beam_clean += [float(editdistance.eval(estimatedWord_beam_clean, name)) / len(name)]
            # print(editdistance.eval(estimatedWord, name))
            wer_path += [
                float(editdistance.eval(estimatedWord_path.split(' '), name.split(' '))) / len(name.split(' '))]
            wer_beam += [
                float(editdistance.eval(estimatedWord_beam.split(' '), name.split(' '))) / len(name.split(' '))]
            wer_beam_clean += [float(editdistance.eval(estimatedWord_beam_clean.split(' '), name.split(' '))) / len(
                name.split(' '))]
            #print('estimated_path: ' + estimatedWord_path + ' ; estimated_beam: ' + estimatedWord_beam + ' ; real: ' + name)


        overall_cer_path = sum(cer_path) / len(cer_path)
        overall_cer_beam = sum(cer_beam) / len(cer_beam)
        overall_cer_beam_clean = sum(cer_beam_clean) / len(cer_beam_clean)
        overall_wer_path = sum(wer_path) / len(wer_path)
        overall_wer_beam = sum(wer_beam) / len(wer_beam)
        overall_wer_beam_clean = sum(wer_beam_clean) / len(wer_beam_clean)

        print('Path CER: ' + str(overall_cer_path))
        print('Beam CER: ' + str(overall_cer_beam))
        print('Beam Clean CER: ' + str(overall_cer_beam_clean))
        print('Path WER: ' + str(overall_wer_path))
        print('Beam WER: ' + str(overall_wer_beam))
        print('Beam Clean WER: ' + str(overall_wer_beam_clean))

        return overall_cer_path, overall_cer_beam, overall_wer_path, overall_wer_beam, overall_cer_beam_clean, overall_wer_beam_clean
    def saveModel(self, filename, model_path = model_path):
        torch.save(self.net.state_dict(), model_path + filename)

    def loadModel(self, filename, model_path = model_path):
        if isfile(model_path + filename):
            load_parameters = torch.load(model_path + filename)
            self.net.load_state_dict(load_parameters)
            self.net.to(device)
            logger.info('Loading model parameters for RD successfull')
        elif filename is not None:
            logger.info('Loading model parameters failed, ' + str(model_path + filename) + 'not found')


# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return np.asarray(sequences[0][0])