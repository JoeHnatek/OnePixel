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

import copy
import random
import numpy as np


SEED = 0
random.seed(SEED)
np.random.seed(SEED)

class PerturbationGrayScale:
    def __init__(self, x, y, gray, dim=32, filename=None, tClass=None, tConf=None, aClass=None, aClassConf=None):

        assert (x >= 0 and x <=
                dim), "Value given for x is not between 0 and dim. Given: {}".format(x)
        assert (y >= 0 and y <=
                dim), "Value given for y is not between 0 and dim. Given: {}".format(y)
        assert (gray >= 0 and gray <=
                255), "Value given for gray is not between 0 and 255. Given: {}".format(gray)

        self.__x = x
        self.__y = y
        self.__gray = gray
        self.__filename = filename
        self.__tClassification = tClass
        self.__tConfidence = tConf
        self.__classification = aClass
        self.__classificationConfidence = aClassConf
        self.__image = None

    @property
    def getX(self):
        return self.__x

    @property
    def getY(self):
        return self.__y

    @property
    def getGray(self):
        return self.__gray

    @property
    def getCoords(self):
        return (self.__x, self.__y)

    @property
    def filename(self):
        return self.__filename

    @property
    def targetClassification(self):
        return self.__tClassification

    @property
    def targetConfidence(self):
        return self.__tConfidence

    @property
    def classification(self):
        return self.__classification

    @property
    def classificationConfidence(self):
        return self.__classificationConfidence

    @property
    def grayscale(self):
        return self.__grayscale_cam

    @property
    def image(self):
        return copy.deepcopy(self.__image)

    @filename.setter
    def filename(self, value):
        self.__filename = value

    @targetClassification.setter
    def targetClassification(self, value):
        self.__tClassification = value

    @targetConfidence.setter
    def targetConfidence(self, value):
        self.__tConfidence = value

    @classification.setter
    def classification(self, value):
        self.__classification = value

    @classificationConfidence.setter
    def classificationConfidence(self, value):
        self.__classificationConfidence = value

    @grayscale.setter
    def grayscale(self, value):
        self.__grayscale_cam = value

    @image.setter
    def image(self, value):
        self.__image = copy.deepcopy(value)

    def __str__(self):
        return """(x,y)\t: ({}, {})\nGray\t: ({})
        \nTarget Class\t: {}\nTarget\t: {}\nClass\t: {}\nClass Confidence\t: {}\n""".format(self.__x, self.__y, self.__gray,
                                                                                            self.__tClassification, self.__tConfidence, self.__classification, self.__classificationConfidence)
def createCandidateSol(dim):
    # Create the gray and xy values
    gray = int(np.random.default_rng().normal(128, 127)) % 256

    x = random.randint(0, dim-1)
    y = random.randint(0, dim-1)

    return PerturbationGrayScale(x, y, gray)

def createChildSol(x1, x2, x3, f, dim):

    x = int(x1.getX + f * (x2.getX - x3.getX)) % dim
    y = int(x1.getY + f * (x2.getY - x3.getY)) % dim
    gray = int(x1.getGray + f * (x2.getGray - x3.getGray)) % 256

    return PerturbationGrayScale(x, y, gray)

def createBestTwoSol(x1, x2, x3, x4, f, best, dim):

    x = int(best.getX + f * (x1.getX - x2.getX) + f * (x3.getX - x4.getX)) % dim
    y = int(best.getY + f * (x1.getY - x2.getY) + f * (x3.getY - x4.getY)) % dim
    gray = int(best.getGray + f * (x1.getGray - x2.getGray) + f * (x3.getGray - x4.getGray)) % 256

    return PerturbationGrayScale(x, y, gray)