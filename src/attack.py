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
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
from networks import allCNN
from networks import NiN
from Perturbation import Perturbation
from torchvision.models import vgg16
from PIL import Image as img

torch.manual_seed(0)


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
            "Value: {}, is not a valid network to attack\nValid networks: allcnn, nin, vgg".format(string))


parser = argparse.ArgumentParser(
    description="Perform one-pixel attacks on images")
parser.add_argument('-data', type=dir_path, default='../cifar-10-kaggle/test')
parser.add_argument('-model', type=network)
parser.add_argument('-n', type=int, default=1)
parser.add_argument('-i', nargs="+", type=int)
args = parser.parse_args()

args = vars(parser.parse_args())
for key in args:
    value = args[key]
    if value == None:
        parser.error("Missing required argument for -{}".format(key))

if len(args['i']) == 1 and args['i'][0] == -1:
    RAND = True
    print("RANDOM MODE ON")
else:
    RAND = False
    assert len(
        args['i']) == args['n'], "Number of images to attack is not equal to the number of indexes given"
    for i in args['i']:
        if i < 1:
            parser.error(
                "Random mode is off - An index given in the -i flag is negative, please remove value: {}".format(i))


def sortDirectory(aList):
    result = []
    for i in aList:
        result.append(int(i.split(".")[0]))  # Split on '.png'

    return sorted(result)


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
    # Create the rgb and xy values
    r = int(numpy.random.default_rng().normal(128, 127)) % 256
    g = int(numpy.random.default_rng().normal(128, 127)) % 256
    b = int(numpy.random.default_rng().normal(128, 127)) % 256

    x = random.randint(0, 32-1)
    y = random.randint(0, 32-1)

    return Perturbation(x, y, r, g, b)


def createChildSol(possiblePerturbations, f=0.5):

    x1, x2, x3 = numpy.random.choice(possiblePerturbations, 3)

    x = int(x1.getX + f * (x2.getX - x3.getX)) % 32
    y = int(x1.getY + f * (x2.getY - x3.getY)) % 32
    r = int(x1.getR + f * (x2.getR - x3.getR)) % 256
    g = int(x1.getG + f * (x2.getG - x3.getG)) % 256
    b = int(x1.getB + f * (x2.getB - x3.getB)) % 256

    return Perturbation(x, y, r, g, b)


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

            possiblePerturbations = listCS[i:]+listCS[:i-1]
            childPerturbation = createChildSol(possiblePerturbations)

            image2 = createImage(srcImage, childPerturbation)

            batch = createSingleBatch(image2)

            con, top = classify(batch, target)

            childPerturbation.filename = imageFilename
            childPerturbation.targetClassification = target
            childPerturbation.targetConfidence = con
            childPerturbation.classification = top[0]
            childPerturbation.classificationConfidence = top[1]

            childPerturbation.image = image2

            parentPerturbation = listCS[i]

            if (childPerturbation.targetConfidence >= 90):
                stop = True
                highest = copy.deepcopy(childPerturbation)

            if childPerturbation.targetConfidence > parentPerturbation.targetConfidence:
                listCS[i] = childPerturbation

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

            with open('results1.txt', 'a') as f:
                f.write("{} {} {:.4f} {} {:.4f} {} {:.4f} {}\n".format(image, oClass, oConf, target, highest.targetConfidence,
                        highest.classification, highest.classificationConfidence, (highest.getCoords, highest.getRGB)))
            del highest
        srcImage.close()


if __name__ == '__main__':

    if args['model'] == 'allcnn':
        net = allCNN.Net()
        PATH = '../models/cifar10-allCNN-20210801T132829.pth'
    elif args['model'] == 'nin':
        net = NiN.Net()
        PATH = '../models/cifar10-NiN-20210315T190616.pth'
    elif args['model'] == 'vgg':
        net = vgg16()
        PATH = '../models/cifar10-VGG16-20210315T233328.pth'

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.cuda.manual_seed_all(0)
    else:
        raise Exception("Need CUDA to run this program.")

    net = nn.DataParallel(net)
    net.to(DEVICE)
    net.load_state_dict(torch.load(PATH), strict=False)

    TRANSFORM = transforms.Compose([transforms.ToTensor()])

    IMAGES = set(args['i']) if not RAND else random.sample(
        range(0, 300000 + 1), args['n'])

    CLASSES = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    CLASS_DICT = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3,
                  'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    PATH = args['data']

    main()
