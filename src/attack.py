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
from networks import medNets
import Perturbation
import PerturbationGrayScale
from torchvision.models import vgg16
from PIL import Image as img
import matplotlib.pyplot as plt
import seaborn as sns
import medmnist
from medmnist import INFO, Evaluator
import settings


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
    networks = ["allcnn", "nin", "vgg", "path", "breast", "pne", "oct", "blood"]

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

def createSingleBatch(aImage):
    if args['data'] == "cifar":
        tmp = transforms.ToTensor()((aImage)).to(DEVICE)
    else:   # For now, leave for medmnist
        trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])
        tmp = trans(aImage).to(DEVICE)
    
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
        return Perturbation.createCandidateSol(IMAGE_DIMENSION)
    elif MODE == 'gray':
        return PerturbationGrayScale.createCandidateSol(IMAGE_DIMENSION)
    else:
        exit() # crash

def createChildSol(possiblePerturbations, f=0.5):

    x1, x2, x3 = np.random.choice(possiblePerturbations, 3)

    if MODE == 'RGB':
        return Perturbation.createChildSol(x1, x2, x3, f, IMAGE_DIMENSION)
    elif MODE == 'gray':
        return PerturbationGrayScale.createChildSol(x1, x2, x3, f, IMAGE_DIMENSION)
    else:
        exit() # crash

def getF():

    return float(np.random.default_rng().normal(0.5, 0.3)) % 2

def createBestTwoSol(possiblePerturbations, best):

    x1, x2, x3, x4 = np.random.choice(possiblePerturbations, 4)

    f = getF()

    if MODE == 'RGB':
        return Perturbation.createBestTwoSol(x1, x2, x3, x4, f, best, IMAGE_DIMENSION)
    elif MODE == 'gray':
        return PerturbationGrayScale.createBestTwoSol(x1, x2, x3, x4, f, best, IMAGE_DIMENSION)
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
            #d[childPerturbation.getX][childPerturbation.getY] += 1
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
        
        if args['data'] == "cifar":
            srcImage = img.open('{}/{}.png'.format(PATH, image))
        else:   # For now, leave for medmnist
            srcImage = medmnistDataset[image][0]

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
                if MODE == "RGB":
                    f.write("{} {} {:.4f} {} {:.4f} {} {:.4f} {}\n".format(image, oClass, oConf, target, highest.targetConfidence,
                            highest.classification, highest.classificationConfidence, (highest.getCoords, highest.getRGB)))
                else:
                    f.write("{} {} {:.4f} {} {:.4f} {} {:.4f} {}\n".format(image, oClass, oConf, target, highest.targetConfidence,
                            highest.classification, highest.classificationConfidence, (highest.getCoords, highest.getGray)))
            del highest
        srcImage.close()


if __name__ == '__main__':
    
    print("Loading settings...", end="", flush=True)

    parser = argparse.ArgumentParser(
    description="Perform one-pixel attacks on images")
    parser.add_argument('-data', type=str, default='cifar')
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

    START_TIMESTAMP = args['t']

    if len(args['i']) == 1 and args['i'][0] == -1:
        RAND = True
    else:
        RAND = False
        assert len(
            args['i']) == args['n'], "Number of images to attack is not equal to the number of indexes given"
        for i in args['i']:
            if i < 0:
                parser.error(
                    "Random mode is off - An index given in the -i flag is negative, please remove value: {}".format(i))


    if args['data'] == "cifar":

        PATH, IMAGES, CLASSES, CLASS_DICT, IMAGE_DIMENSION = settings.cifar(args['i'], RAND)
    
    else:

        func = getattr(settings, args['data'])
        DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION = func(args['i'], RAND)

    MODEL = args['model']
    MODE = args['mode']
    HOST = socket.gethostname()
    
    net, MODEL_PATH = settings.getModel(MODEL)

    try:
        DEVICE = torch.device("cuda")
    except:
        raise Exception("Need CUDA to run this program.")

    if args['data'] == 'cifar':
        net = nn.DataParallel(net)
        net.to(DEVICE)
        net.load_state_dict(torch.load(MODEL_PATH), strict=False)
    else:
        net.to(DEVICE)
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['net'], strict=True)

    TRANSFORM = transforms.Compose([transforms.ToTensor()])

    #d = np.zeros((28,28))

    RESULTS_PATH = '../results/{}/'.format(START_TIMESTAMP)

    try:
        os.mkdir(os.path.dirname(RESULTS_PATH))
    except:
        try:
            os.path.exists(os.path.dirname(RESULTS_PATH))
        except:
            raise NotADirectoryError
    
    with open(RESULTS_PATH+"{}.txt".format(HOST), 'a') as f:
        f.write("MODEL: {}\n".format(MODEL))
        f.write("MODEL_PATH: {}\n".format(MODEL_PATH))
        f.write("HOST: {}\n".format(HOST))
        f.write("SEED: {}\n".format(SEED))
        for i in IMAGES:
            if args['data'] == "cifar":
                f.write("IMAGE: {}.png\n".format(i))
            else:
                f.write("IMAGE: {}\n".format(i))
    print("Loaded\nRunning attacks.")
    main()
    #ax = sns.heatmap(d, robust=True, cmap='rainbow', yticklabels=False, xticklabels=False)
    #ax.tick_params(left=False, bottom=False)
    #plt.savefig("breast-hm.png")