import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, im_size, n_classes):
        """
        Initialization of the softmax classifier on images.
        Arguments:
            im_size: a tuple of ints of (n_channel, height, width)
            n_classes: int of number of classes
        """
        super().__init__()  # if in Python2, need to be changed to "super(Softmax, self).__init__()"
        in_dim = np.prod(im_size)  # input dimension
        self.fc = nn.Linear(in_dim, n_classes)  # fully-connected layer

    def forward(self, images):
        """
        Forward pass of the softmax classifier on images to output scores of each sample.
        Arguments:
            images: a tensor of size (N, C, H, W), whichi stands for (batch_size, num_channel, height, width)
        Outputs:
            scores: a Torch variable of size (N, n_classes) indicating the scores for each sample in each class
        """
        images = images.view(images.shape[0], -1)  # reshape image tensor
        scores = self.fc(images)
        return scores


class Two_Layer_NN(nn.Module):
    def __init__(self, im_size, dim_hidden, n_classes):
        """
        Initialization of the two-layer neural network classifier on images.
        Arguments:
            im_size: a tuple of ints of (n_channel, height, width)
            dim_hidden: int of dimension of the hidden layer
            n_classes: int of number of classes
        """
        super().__init__()  # if in Python2, need to be changed to "super(Two_Layer_NN, self).__init__()"
        in_dim = int(np.prod(im_size))  # dimension of the input tensor
        self.fc1 = nn.Linear(in_dim, dim_hidden)  # fully-connected layer
        self.fc2 = nn.Linear(dim_hidden, n_classes)  # fully-connected layer

    def forward(self, images):
        """
        Forward pass of the two-layer neural network classifier on images to output scores of each sample.
        Arguments:
            images: a tensor of size (N, C, H, W), which stands for (batch_size, num_channel, height, width)
        Outputs:
            scores: a Torch variable of size (N, n_classes) indicating the scores for each sample in each class
        """
        images = images.view(images.shape[0], -1)  # reshape the tensor
        scores = self.fc1(images)
        scores = F.relu(scores)  # Relu function
        scores = self.fc2(scores)
        return scores


class Conv_Net(nn.Module):
    def __init__(self, im_size, dim_hidden, n_classes, kernel_size):
        """
        Initialization of the two-layer neural network classifier on images.
        Arguments:
            im_size: a tuple of ints of (n_channel, height, width)
            dim_hidden: int of dimension of the hidden layer
            n_classes: int of number of classes
            kernel_size: int of the kernel size
        """
        super().__init__()  # if in Python2, need to be changed to "super(Two_Layer_NN, self).__init__()"
        in_channels, in_height, in_width = im_size
        self.conv2d = nn.Conv2d(in_channels, dim_hidden, kernel_size)  # convolution layer
        hidden_height = in_height - kernel_size + 1
        hidden_width = in_width - kernel_size + 1
        hidden_dim = np.prod([hidden_height, hidden_width, dim_hidden])
        self.fc = nn.Linear(hidden_dim, n_classes)  # fully-connected layer

    def forward(self, images):
        """
        Forward pass of the convolutional neural network classifier on images to output scores of each sample.
        Arguments:
            images: a tensor of size (N, C, H, W), which stands for (batch_size, num_channel, height, width)
        Outputs:
            scores: a Torch variable of size (N, n_classes) indicating the scores for each sample in each class
        """
        scores = self.conv2d(images)
        scores = F.relu(scores)  # Relu function
        scores = scores.view(images.shape[0], -1)  # reshape image tensor
        scores = self.fc(scores)
        return scores
