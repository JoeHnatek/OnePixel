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

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Conv2d(input, output, kernal, stride)
        self.conv1 = nn.Conv2d(3, 192, 5, padding=1)
        self.conv2 = nn.Conv2d(192, 160, 1, padding=1)
        self.conv3 = nn.Conv2d(160, 96, 1, padding=1)
        self.conv4 = nn.Conv2d(96, 192, 5, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 5, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 5, padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)

        self.maxPool = nn.MaxPool2d((3, 3), 2)

        self.avgPool3 = nn.AvgPool2d((3, 3), 1)
        self.avgPool8 = nn.AvgPool2d((8, 8), 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.maxPool(x)

        x = F.dropout(x, 0.5, training=self.training)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = self.avgPool3(x)

        x = F.dropout(x, 0.5, training=self.training)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.conv9(x)

        x = self.avgPool8(x)

        x = nn.Flatten(1)(x)

        return x
