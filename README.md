# Simple Neural Networks written in PyTorch
## Description:

This repository includes Softmax, two-layer NN and simple CNN image classifiers written in PyTorch in Python3. This would be good starting point for new PyTorch users to learn how to construct models and how to train them using PyTorch.

## Model Details:
The models.py includes three different models, including Softmax, two-layer NN and simple CNN image classifiers. The train.py trains one of the models with parameter specifying detailed settings. And the cifar10.py is used to download or verify the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) as the training dataset in this repository.

## Model Performance:
The training and validation loss and validation accuracy curves are shown below for the three models in 10 epochs.

![alt text](/images/softmax.png)

![alt text](/images/twolayernn.png)

![alt text](/images/convnet.png)

For the test set, the above three models achieve accuracies of 28.09%, 40.20%, 44.77%, respectively with 10 epochs of training.

## Code Usage
1. If running shell, directly run "softmax.sh"/"twolayernn.sh"/"convnet.sh". The CIFAR-10 dataset will be automatically downloaded into a folder named "data" in the current path if necessary. And the model will be trained for 1 epoch and saved in a ".pt" file and the logs saved in a ".log" file. The curve will appear when the training is finished.
2. If running the script in IDEs like PyCharm, simply add the parameters before training the model. PyCharm for example, go to Run -> Edit Configurations... -> Parameters, and copy the parameters from the corresponding .sh file in. Then run train.py to train the model or for further debugging.

## Reference
1. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/).
2. [https://github.com/pytorch/examples/blob/master/mnist/main.py](https://github.com/pytorch/examples/blob/master/mnist/main.py).
3. [https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py](https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py)
