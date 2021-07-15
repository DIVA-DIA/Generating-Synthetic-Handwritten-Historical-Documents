import csv
import math
import random
import time

import numpy as np
import torch
import cv2
from torch.autograd import Variable
from torchvision.transforms import transforms

from HTR_ctc.utils.auxilary_functions import image_resize
from .generated_config import *
from .models import Generator


bounding_box = 256
delimiter = '$'
# Picture crop:
#  __________
# |x0y0     |
# |         |
# |    x1,y1|
#  _________

def getWordsInCrop(name, source, x0, y0, x1, y1, source_dataset):

    scope_tolerance = bounding_box * 0.01 # normally try 0.05
    # print('x0: ' + str(x0) + ' y0: ' + str(y0) + 'x1: ' + str(x1) + ' y1: ' + str(y1))
    csvCrop = open(source + 'csv-crop/' + name + '-crop.csv', 'w+')
    word = ""
    pre = ""
    post = ""
    y_scope_out = False
    x0_new = 0
    y0_new = 0
    x1_new = 0
    y1_new = bounding_box
    csvCropString = []
    wordOver = False
    fix = False
    hasWord = False # Variable that is returned determinating if the crop contain a word (It could also be a crop thats is only a lettrine)
    csvCrop.write(
        'word' + delimiter + 'x0' + delimiter + 'y0' + delimiter + 'x1' + delimiter + 'y1' + delimiter + 'y_scope_out' + delimiter + 'pre' + delimiter + 'post\n')
    with open(data_path + source_dataset + '-csv/' + name + '.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            wordOver = False
            fix = False
            if row['char'] == ' ' or y1_new < float(row['y1']):  # word over
                if (word != "" or pre != "" or post != "") and pfloat(x0_new - x0) != 256 and pfloat(x1_new - x0) != 0: # while the last two and statements may seem obsolute, otherwise there is a bug where a random M occurs in the csv that has no length
                    hasWord = True
                    csvCrop.write(
                        word + delimiter + str(pfloat(x0_new - x0)) + delimiter + str(pfloat(y0_new - y0)) + delimiter + str(
                            pfloat(x1_new - x0)) + delimiter + str(pfloat(y1_new - y0)) + delimiter + str(
                            y_scope_out) + delimiter + pre + delimiter + post + '\n')
                    csvCropString.append(word + delimiter + str(pfloat(x0_new - x0)) + delimiter + str(pfloat(y0_new - y0)) + delimiter + str(
                            pfloat(x1_new - x0)) + delimiter + str(pfloat(y1_new - y0)) + delimiter + str(
                            y_scope_out) + delimiter + pre + delimiter + post)
                word = ""
                pre = ""
                post = ""
                y_scope_out = False
                x0_new = 0
                y0_new = 0
                x1_new = 0
                y1_new = bounding_box
                if row['char'] == ' ':
                    wordOver = True
            if x0 - scope_tolerance < float(row['x0']) and y0 - scope_tolerance < float(row['y0']) and x1 + scope_tolerance > float(row['x1']) and y1 + scope_tolerance > float(row['y1']) and not wordOver:
                if x0_new == 0:
                    x0_new = float(row['x0'])
                if y0_new == 0:
                    y0_new = float(row['y0'])
                    y1_new = float(row['y1'])
                if x1_new < float(row['x1']):
                    x1_new = float(row['x1'])

                if x0 < float(row['x0']) and y0 < float(row['y0']) and x1 > float(row['x1']) and y1 > float(row['y1']):
                    word = word + row['char']
                else:
                    if x0 - scope_tolerance < float(row['x0']) and not (x0 < float(row['x0'])): # not has to be checked extra, because otherwise it would also be true when y with tolerance would be true (last if)
                        pre = pre + row['char']
                        fix = True
                    if x1 + scope_tolerance > float(row['x1']) and not (x1 > float(row['x1'])):
                        post = post + row['char']
                        fix = True
                    if (y0 - scope_tolerance < float(row['y0']) and not (y0 < float(row['y0']))) or (y1 + scope_tolerance > float(row['y1'])  and not (y1  > float(row['y1']))):
                        if not fix:  # only add to word if char is not pre of postfix
                            word = word + row['char']
                        y_scope_out = True

    csvCrop.close()
    return hasWord, csvCropString

def cropWords(image, name='', source='', csvfile=None):
    #print(name)
    if torch.is_tensor(image):
        image = image.detach().squeeze(0).transpose(0,2).transpose(0,1).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, :, np.newaxis]

    word_array = []
    info_array = []

    if csvfile is None:
        csvfile = open(source + 'csv-crop/' + name + '.csv')
        #with open(source + 'csv-crop/' + name + '.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            # print(row['y0'])
            #todo: fix word positions in pdftocrop
            y0 = math.floor(float(row['y0']) + -10)
            if y0 < 0:
                y0 = 0
            output = image[y0 : math.ceil(float(row['y1']) + 10), math.floor(float(row['x0'])) : math.ceil(float(row['x1'])), : ]
            word_array.append(output)
            info_array.append(row['pre'] + row['word'] + row['post'])
            # todo: insert below code and adjust Reading Discriminator for pre and postfixes
            #info_array.append({'word': row['word'], 'pre': row['pre'], 'post': row['post'], 'y_scope_out': row['y_scope_out']})

        #for x in range(0, len(word_array)):
        #    toimage((word_array[x].numpy()), cmin=-1, cmax=1).save('generate_book-crop/(' + info_array[x]['pre'] + ')' + info_array[x]['word'] + '(' + info_array[x]['post'] + ')' + info_array[x]['y_scope_out']  +'.png')
    else:
        for rows in csvfile:
            row = ''.join(rows)
            row = row.rsplit('$')
            # 0 : word , 1: x0 , 2: y0 , 3: x1 , 4 : y1 , 5 : y_scope_out, 6 : pre , 7 : post
            y0 = math.floor(float(row[2]) + -10)
            if y0 < 0:
                y0 = 0
            output = image[y0: math.ceil(float(row[4]) + 10),
                     math.floor(float(row[1])): math.ceil(float(row[3])), :]
            word_array.append(output)
            info_array.append(row[6] + row[0] + row[7])

    return word_array, info_array

def pfloat(number): # positive float
    if number <= 0:
        return 0
    elif number >= bounding_box:
        return bounding_box
    else:
        return number



def getRandomCrop(image, image_name, source, source_dataset):
    boundingBox_size = 256
    hasWords = False
    while True:
        rand_x = random.randint(600,
                                2154)  # the text of the document is between 600 and 2154 (-256 = 2510) width
        rand_y = random.randint(582, 2937)
        # print('x: ' + str(rand_x) + ', y: ' + str(rand_y))
        #image = tf.image.crop_to_bounding_box(image, rand_y, rand_x, boundingBox_size, boundingBox_size)
        croppedImage = image[rand_y: rand_y + boundingBox_size, rand_x: rand_x + boundingBox_size,:]
        hasWords, csvCropString = getWordsInCrop(image_name.rsplit('.')[0], source, rand_x, rand_y, rand_x + boundingBox_size,
                                       rand_y + boundingBox_size, source_dataset)
        if np.mean(croppedImage) < 230 and hasWords:  # recrop if picture is too white (not enough text)
            break

    return croppedImage, csvCropString

def getCrop(image, x_axis, y_axis):
    boundingBox_size = 256

    croppedImage = image[y_axis: y_axis + boundingBox_size, x_axis: x_axis + boundingBox_size,:]


    return croppedImage

#todo: delete
def normalize_array(array):
    return np.subtract(np.divide(array, 127.5), 1)

def resizeImg(img, fixed_size , order=1):
        nheight = fixed_size[0]
        nwidth = fixed_size[1]
        if nheight is None:
            nheight = img.shape[0]
        if nwidth is None:
            nwidth = int(np.random.uniform(.8, 1.2) * img.shape[1] * nheight / img.shape[0])

        img = image_resize(img, height=nheight-16, width=nwidth, order=order)
       # img = centered(img, (nheight, int(1.2 * nwidth) + 32))
        img = torch.Tensor(img).float().unsqueeze(0)
        return img

def generateCrops(nr_of_channels, source, just_generate = False, crop_path = 'train/A/', normalize=True, generator_path='',  source_dataset = 'EG-BG-LC'):
    print('Creating new crops')
    data = []
    image_count = 0
    transforms_ = [transforms.ToTensor()]
    transform = transforms.Compose(transforms_)

    if generator_path != '':
        Tensor = torch.cuda.FloatTensor
        input_A = Tensor(1, 3, 256, 256)
        netG_A2B = Generator(3, 3)
        netG_A2B.cuda()
        netG_A2B.load_state_dict(torch.load(generator_path))

    all_image_names = os.listdir(data_path + source_dataset)
    for image_name in all_image_names:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = cv2.imread(os.path.join(data_path + source_dataset, image_name), cv2.IMREAD_GRAYSCALE)
                # todo: change data loader to delete one dimension in black and white, then delete squeeze in get
                if not just_generate:
                    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                image = image[:, :, np.newaxis]
            else:  # RGB image -> street view
                image = cv2.imread(os.path.join(data_path + source_dataset, image_name))
                if not just_generate:
                    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


            image, csvCropString = getRandomCrop(image, image_name, source, source_dataset)
            if generator_path != '':
                real_A = Variable(input_A.copy_(transform(image)))
                image = (netG_A2B(real_A).data)

            if just_generate:
                data.append([image.copy(), csvCropString])
            else:
                word_array, info_array = cropWords(image, image_name.rsplit('.')[0] + '-crop',
                                                   source)
                for i in range(0, len(word_array)):
                    if len(word_array[i].shape) == 3:
                        _, w, _ = word_array[i].shape
                        if w > 1:
                            data.append([word_array[i].copy(), info_array[i]])
            t3 = time.time()
        image_count += 1
        if image_count % (len(data_image_names) // 10) == 1:
            print(str(image_count) + '/' + str(len(data_image_names)))

    print('Finished crops')
    return data
    
