"""
MIT License

Copyright (c) 2022 Joe Hnatek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import argparse
import copy
import os
import numpy as np
import socket
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
from datetime import datetime
from networks import allCNN
from networks import NiN
from networks import vgg
import Perturbation
import PerturbationGrayScale
from torchvision.models import vgg16
from PIL import Image as img
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def dir_path(string):
    if os.path.exists(string):
        return string
    else:
        raise NotADirectoryError(string)


def network(string):
    networks = ["allcnn", "nin", "vgg"]

    if string in networks:
        return string
    else:
        parser.error(
            f"Value: {string}, is not a valid network to attack\nValid networks: allcnn, nin, vgg")

def mode(string):
    modes = ["RGB", "gray"]

    if string in modes:
        return string
    else:
        parser.error(
            f"Value: {string}, is not a valid mode to attack\nValid modes: RGB, gray")

parser = argparse.ArgumentParser(
    description="Perform one-pixel attacks on images")
parser.add_argument('-data', type=dir_path, default='../cifar-10-kaggle/test')
parser.add_argument('-model', type=network)
parser.add_argument('-mode', type=mode)
parser.add_argument('-n', type=int, default=1)
parser.add_argument('-i', nargs="+", type=int)
parser.add_argument('-t', type=str, default=datetime.now().strftime("%Y%m%dT%H%M%S"))
args = parser.parse_args()

args = vars(parser.parse_args())
for key in args:
    value = args[key]
    if value == None:
        parser.error("Missing required argument for -{}".format(key))

if len(args['i']) == 1 and args['i'][0] == -1:
    RAND = True
else:
    RAND = False
    assert len(
        args['i']) == args['n'], "Number of images to attack is not equal to the number of indexes given"
    for i in args['i']:
        if i < 1:
            parser.error(
                "Random mode is off - An index given in the -i flag is negative, please remove value: {}".format(i))

START_TIMESTAMP = args['t']


def createSingleBatch(aImage):
    tmp = transforms.ToTensor()((aImage)).to(DEVICE)
    singleBatch = tmp.unsqueeze_(0)
    return singleBatch


def classify(batch, target=None):
    """
    Target allows us to 'target' a specific classification.
    Otherwise, get highest confidence.
    """
    net.eval()
    with torch.torch.inference_mode():
        output = net(batch)
        _, pClass = torch.max(output, 1)
        sm = torch.nn.functional.softmax(output[0], dim=0)
        classification = CLASSES[pClass[0]]
        if target:
            confidence = sm[CLASS_DICT[target]].item() * 100
            top = (classification, sm.topk(1, dim=0)[0].item() * 100)
        else:
            confidence = sm.topk(1, dim=0)[0].item() * 100

    if target:
        return confidence, top
    else:
        return classification, confidence


def createCandidateSol():
    
    if MODE == 'RGB':
        return Perturbation.createCandidateSol()
    elif MODE == 'gray':
        return PerturbationGrayScale.createCandidateSol()
    else:
        exit() # crash

def createChildSol(possiblePerturbations, f=0.5):

    x1, x2, x3 = np.random.choice(possiblePerturbations, 3)

    if MODE == 'RGB':
        return Perturbation.createChildSol(x1, x2, x3, f)
    elif MODE == 'gray':
        return PerturbationGrayScale.createChildSol(x1, x2, x3, f)
    else:
        exit() # crash

def getF():

    return float(np.random.default_rng().normal(0.5, 0.3)) % 2

def createBestTwoSol(possiblePerturbations, best):

    x1, x2, x3, x4 = np.random.choice(possiblePerturbations, 4)

    f = getF()

    if MODE == 'RGB':
        return Perturbation.createBestTwoSol(x1, x2, x3, x4, f, best)
    elif MODE == 'gray':
        return PerturbationGrayScale.createBestTwoSol(x1, x2, x3, x4, f, best)
    else:
        exit() # crash

def createImage(image, p):
    image = copy.deepcopy(image)
    image.putpixel(p.getCoords, p.getRGB)

    return image


def attackDE(image, target, imageFilename, f=0.5, population=400):

    srcImage = copy.deepcopy(image)
    MAX_ITER = 100
    highest = None

    listCS = [createCandidateSol() for i in range(population)]

    # Initialize our first group of parents

    for p in listCS:

        image = createImage(srcImage, p)

        batch = createSingleBatch(image)

        con, top = classify(batch, target)

        p.filename = imageFilename
        p.targetClassification = target
        p.targetConfidence = con
        p.classification = top[0]
        p.classificationConfidence = top[1]
        p.image = image

        if highest != None:
            if p.targetConfidence > highest.targetConfidence:
                highest = copy.deepcopy(p)
        else:
            highest = copy.deepcopy(p)

    if highest.targetConfidence >= 90:
        return highest

    a = 0
    stop = False
    while a < MAX_ITER and stop == False:
        for i in range(population):

            possiblePerturbations = listCS[i+1:]+listCS[:i]
            childPerturbation = createChildSol(possiblePerturbations)#, highest)

            image2 = createImage(srcImage, childPerturbation)

            batch = createSingleBatch(image2)

            con, top = classify(batch, target)

            childPerturbation.filename = imageFilename
            childPerturbation.targetClassification = target
            childPerturbation.targetConfidence = con
            childPerturbation.classification = top[0]
            childPerturbation.classificationConfidence = top[1]
            
            childPerturbation.image = image2
            d[childPerturbation.getX][childPerturbation.getY] += 1
            parentPerturbation = copy.deepcopy(listCS[i])

            if (childPerturbation.targetConfidence >= 90):
                stop = True
                highest = copy.deepcopy(childPerturbation)

            if childPerturbation.targetConfidence > parentPerturbation.targetConfidence:
                listCS[i] = copy.deepcopy(childPerturbation)

            if (childPerturbation.targetConfidence > highest.targetConfidence):
                highest = copy.deepcopy(childPerturbation)

        a += 1
    for p in listCS:
        del p

    return highest


def main():

    for image in IMAGES:

        srcImage = img.open('{}/{}.png'.format(PATH, image))

        # Create a single batch for the source image
        batch = createSingleBatch(srcImage)
        # Get the original classification and confidence of the source image
        oClass, oConf = classify(batch)

        for target in CLASSES:
            if target == oClass:
                continue
            highest = attackDE(srcImage, target, image)
            
            dir_path(RESULTS_PATH)

            with open(RESULTS_PATH+"{}.txt".format(HOST), 'a') as f:
                f.write("{} {} {:.4f} {} {:.4f} {} {:.4f} {}\n".format(image, oClass, oConf, target, highest.targetConfidence,
                        highest.classification, highest.classificationConfidence, (highest.getCoords, highest.getRGB)))
            del highest
        srcImage.close()


if __name__ == '__main__':

    MODEL = args['model']
    MODE = args['mode']
    HOST = socket.gethostname()
    
    if MODEL == 'allcnn':
        net = allCNN.Net()
        MODEL_PATH = '../models/cifar10-allCNN-20210801T132829.pth'
    elif MODEL == 'nin':
        net = NiN.Net()
        MODEL_PATH = '../models/cifar10-NiN-20210315T190616.pth'
    elif MODEL == 'vgg':
        net = vgg16()
        MODEL_PATH = '../models/cifar10-VGG16-20210315T233328.pth'
    elif MODEL == 'fer':
        net = vgg.Vgg()
        MODEL_PATH = '../models/VGGNet'

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        raise Exception("Need CUDA to run this program.")

    net = nn.DataParallel(net)
    net.to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH), strict=False)

    TRANSFORM = transforms.Compose([transforms.ToTensor()])

    IMAGES = set(args['i']) if not RAND else random.sample(
        range(0, 300000 + 1), args['n'])

    CLASSES = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    CLASS_DICT = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3,
                  'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    PATH = args['data']

    d = np.zeros((32,32))

    RESULTS_PATH = '../results/{}/'.format(START_TIMESTAMP)

    try:
        os.mkdir(os.path.dirname(RESULTS_PATH))
    except:
        raise NotADirectoryError
    
    with open(RESULTS_PATH+"{}.txt".format(HOST), 'a') as f:
        f.write("MODEL: {}\n".format(MODEL))
        f.write("MODEL_PATH: {}\n".format(MODEL_PATH))
        f.write("HOST: {}\n".format(HOST))
        f.write("SEED: {}\n".format(SEED))
        for i in IMAGES:
            f.write("IMAGE: {}.png\n".format(i))
    print("now")
    main()
    ax = sns.heatmap(d, robust=True, cmap='rainbow')
    plt.savefig("test3.png")