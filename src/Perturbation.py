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


class Perturbation:
    def __init__(self, x, y, r, g, b, filename=None, tClass=None, tConf=None, aClass=None, aClassConf=None):

        assert (x >= 0 and x <=
                31), "Value given for x is not between 0 and 31. Given: {}".format(x)
        assert (y >= 0 and y <=
                31), "Value given for y is not between 0 and 31. Given: {}".format(y)
        assert (r >= 0 and r <=
                255), "Value given for r is not between 0 and 255. Given: {}".format(r)
        assert (g >= 0 and g <=
                255), "Value given for g is not between 0 and 255. Given: {}".format(g)
        assert (b >= 0 and b <=
                255), "Value given for b is not between 0 and 255. Given: {}".format(b)

        self.__x = x
        self.__y = y
        self.__r = r
        self.__g = g
        self.__b = b
        self.__filename = filename
        self.__tClassification = tClass
        self.__tConfidence = tConf
        self.__classification = aClass
        self.__classificationConfidence = aClassConf
        self.__grayscale_cam = None
        self.__image = None

    @property
    def getX(self):
        return self.__x

    @property
    def getY(self):
        return self.__y

    @property
    def getR(self):
        return self.__r

    @property
    def getG(self):
        return self.__g

    @property
    def getB(self):
        return self.__b

    @property
    def getCoords(self):
        return (self.__x, self.__y)

    @property
    def getRGB(self):
        return (self.__r, self.__g, self.__b)

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
        return """(x,y)\t: ({}, {})\nRGB\t: ({}, {}, {})
        \nTarget Class\t: {}\nTarget\t: {}\nClass\t: {}\nClass Confidence\t: {}\n""".format(self.__x, self.__y, self.__r, self.__g, self.__b,
                                                                                            self.__tClassification, self.__tConfidence, self.__classification, self.__classificationConfidence)
