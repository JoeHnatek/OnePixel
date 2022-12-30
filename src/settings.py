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

from networks import allCNN
from networks import NiN
from networks import vgg
from networks import medNets
import medmnist
from medmnist import INFO, Evaluator
import torchvision.transforms as transforms
import random

def cifar(images, RAND):

    PATH = "../cifar-10-kaggle/test"
    IMAGES = set(images) if not RAND else random.sample(
        range(0, 300000 + 1), args['n'])
    CLASSES = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    CLASS_DICT = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3,
                  'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    IMAGE_DIMENSION = 32

    return PATH, IMAGES, CLASSES, CLASS_DICT, IMAGE_DIMENSION

def path(images, RAND):

    DATASET_INFO = INFO["pathmnist"]
    IMAGES = set(images) if not RAND else random.sample(
        range(0, DATASET_INFO["n_samples"]["test"] + 1), args['n'])
    CLASSES = (
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth_muscle",
            "normal_colon_mucosa",
            "cancer-associated_stroma",
            "colorectal_adenocarcinoma_epithelium")

    CLASS_DICT = {"adipose": 0,
            "background": 1,
            "debris": 2,
            "lymphocytes": 3,
            "mucus": 4,
            "smooth_muscle": 5,
            "normal_colon_mucosa": 6,
            "cancer-associated_stroma": 7,
            "colorectal_adenocarcinoma_epithelium": 8}
    DataClass = getattr(medmnist, "PathMNIST")

    data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    medmnistDataset = DataClass(split='test', transform=None, download=True, as_rgb=False)
    IMAGE_DIMENSION = 28

    return DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION

def breast(images, RAND):

    DATASET_INFO = INFO["breastmnist"]
    IMAGES = set(images) if not RAND else random.sample(
        range(0, DATASET_INFO["n_samples"]["test"] + 1), args['n'])
    CLASSES = (
            "malignant",
            "normal, benign",
            )

    CLASS_DICT = {"malignant": 0,
            "normal, benign": 1}
    DataClass = getattr(medmnist, "BreastMNIST")

    data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    medmnistDataset = DataClass(split='test', transform=None, download=True, as_rgb=True)
    IMAGE_DIMENSION = 28

    return DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION

def pne(images, RAND):
    
    DATASET_INFO = INFO["pneumoniamnist"]
    IMAGES = set(images) if not RAND else random.sample(
        range(0, DATASET_INFO["n_samples"]["test"] + 1), args['n'])
    CLASSES = (
            "normal",
            "pneumonia",
            )

    CLASS_DICT = {"normal": 0,
            "pneumonia": 1}
    DataClass = getattr(medmnist, "PneumoniaMNIST")

    data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    medmnistDataset = DataClass(split='test', transform=None, download=True, as_rgb=True)
    IMAGE_DIMENSION = 28

    return DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION

def oct(images, RAND):
    
    DATASET_INFO = INFO["octmnist"]
    IMAGES = set(images) if not RAND else random.sample(
        range(0, DATASET_INFO["n_samples"]["test"] + 1), args['n'])
    CLASSES = (
            "choroidal neovascularization",
            "diabetic macular edema",
            "drusen",
            "normal"
            )

    CLASS_DICT = {"choroidal neovascularization":0,
            "diabetic macular edema":1,
            "drusen":2,
            "normal":3}
    DataClass = getattr(medmnist, "OCTMNIST")

    data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    medmnistDataset = DataClass(split='test', transform=None, download=True, as_rgb=True)
    IMAGE_DIMENSION = 28

    return DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION

def blood(images, RAND):
    
    DATASET_INFO = INFO["bloodmnist"]
    IMAGES = set(images) if not RAND else random.sample(
        range(0, DATASET_INFO["n_samples"]["test"] + 1), args['n'])
    CLASSES = (
            "basophil",
            "eosinophil",
            "erythroblast",
            "immature_granulocytes(myelocytes,_metamyelocytes_and_promyelocytes)",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet"
            )

    CLASS_DICT = {"basophil":0,
            "eosinophil":1,
            "erythroblast":2,
            "immature_granulocytes(myelocytes,_metamyelocytes_and_promyelocytes)":3,
            "lymphocyte":4,
            "monocyte":5,
            "neutrophil":6,
            "platelet":7}
    DataClass = getattr(medmnist, "BloodMNIST")

    data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    medmnistDataset = DataClass(split='test', transform=None, download=True, as_rgb=True)
    IMAGE_DIMENSION = 28

    return DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION


def getModel(modelName):

    if modelName == 'allcnn':
        net = allCNN.Net()
        MODEL_PATH = '../models/cifar10-allCNN-20210801T132829.pth'
    elif modelName == 'nin':
        net = NiN.Net()
        MODEL_PATH = '../models/cifar10-NiN-20210315T190616.pth'
    elif modelName == 'vgg':
        net = vgg16()
        MODEL_PATH = '../models/cifar10-VGG16-20210315T233328.pth'
    elif modelName == 'fer':
        net = vgg.Vgg()
        MODEL_PATH = '../models/VGGNet'
    elif modelName == "path":
        net = medNets.ResNet18(in_channels=DATASET_INFO["n_channels"], num_classes=9)
        MODEL_PATH = "../models/resnet18_28_1-path.pth"
    elif modelName == "breast":
        net = medNets.ResNet18(in_channels=3, num_classes=2)
        MODEL_PATH = "../models/resnet18_28_1-breast.pth"
    elif modelName == "pne":
        net = medNets.ResNet18(in_channels=3, num_classes=2)
        MODEL_PATH = "../models/resnet18_28_1-pne.pth"
    elif modelName == "oct":
        net = medNets.ResNet18(in_channels=3, num_classes=4)
        MODEL_PATH = "../models/resnet18_28_1-oct.pth"
    elif modelName == "blood":
        net = medNets.ResNet18(in_channels=3, num_classes=8)
        MODEL_PATH = "../models/resnet18_28_1-blood.pth"

    return net, MODEL_PATH

def getMedMNIST(data):

    if data == "path":

        DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION = settings.path(args['i'], RAND)
        
    elif data == 'breast':

        DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION = settings.breast(args['i'], RAND)

    elif data == 'pne':

        DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION = settings.pne(args['i'], RAND)

    elif data == 'oct':

        DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION = settings.oct(args['i'], RAND)

    elif data == 'blood':

        DATASET_INFO, IMAGES, CLASSES, CLASS_DICT, DataClass, data_transform, medmnistDataset, IMAGE_DIMENSION = settings.blood(args['i'], RAND)