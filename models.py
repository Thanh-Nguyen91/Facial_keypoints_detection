## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32,64,2)           
        self.conv3 = nn.Conv2d(64,128,2)        
        self.conv4 = nn.Conv2d(128,256,2)
        self.conv5 = nn.Conv2d(256,512,2)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.drop = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(512*6*6,1000)
        self.fc2 = nn.Linear(1000,136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # Input 1x224x224 => 32x222x222 => 32x111x111
        x =  self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        
        # 32x111x111 => 64x110x110 => 64x55x55
        x =  self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        
        # 64x55x55 => 128x54x54 => 128x27x27
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop(x)
        
        # 128x27x27 => 256x26x26 => 256x13x13
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop(x)
        
        # 256x13x13 => 512x12x12 => 512x6x6
        x = self.pool(F.relu(self.conv5(x)))
        x = self.drop(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
