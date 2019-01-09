# Two Layer Neural Network Classifier
## Description:

A naive implementation of convulutional neural network layers written in Python3.

## Model Details:
The layer.py includes several different layers with both forward and backward pass. Specifically, the layers include fully-connected layer, ReLU function layer, convolution layer, max pooling layer and dropout layer.

### Fully-connected layer ###
The fully-connected layer uses a simple matrix operation of

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;out=Wx&plus;b" title="out=Wx+b" />
</p>

where *W* is the weight matrix, *x* is the input matrix and *b* is the bias matrix. *W* typically has a shape of (*D*, *B*) with *D* as number of dimensions. *x* is usually a matrix of RBG image dataset with shape(*N*, *d*<sub>1</sub>,..., *d*<sub>c</sub>) with *N* as number of images and *d*<sub>1</sub>,..., *d*<sub>*c*</sub> as *c* channel pixel values. 

The back propagation of the fully-connected layer is expressed as follows:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;L}{\partial&space;W}=\frac{\partial&space;L}{\partial&space;Y}X^T" title="\frac{\partial L}{\partial W}=\frac{\partial L}{\partial Y}X^T" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;L}{\partial&space;X}=W^T&space;\frac{\partial&space;L}{\partial&space;Y}" title="\frac{\partial L}{\partial X}=W^T \frac{\partial L}{\partial Y}" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;L}{\partial&space;b_j}=\sum_{b=1}^{B}&space;\frac{\partial&space;L}{\partial&space;y_j}" title="\frac{\partial L}{\partial b_j}=\sum_{b=1}^{B} \frac{\partial L}{\partial y_j}" />
</p>

### ReLU function layer ###
The ReLU function layer simply eliminates all of the negative values of the input and the back propagation only keeps the passed gradient with non-negative values of the input.

### Convolution layer ###
The concolution layer is the key to the CNN architecture. Here a straightforward naive implementation is implemented using loops of loops. 2D convolution is performed on every image and every channel of each image. The convolution matrices are added together for different channels. There are two important parameters: padding and stride, which are the basic concepts of CNN. The forward pass uses the following expression as a single convolution:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;out=\sum_{x_n=1}^{N}\sum_{f_n=1}^{F}conv2d(\sum_{c_n=1}^{c}sum(x_{region}*filter))" title="out=\sum_{x_n=1}^{N}\sum_{f_n=1}^{F}conv2d(\sum_{c_n=1}^{c}sum(x_{region}*filter))" />
</p>

where *conv2d()* is the 2D convolution which have filter sliding across each channel of the image with the specified padding and stride values. And it can be expressed as:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;y[l,c]=\sum_{a=1}^{k_1}&space;\sum_{b=1}^{k_2}x[l&plus;a,c&plus;b]w[a,b]" title="y[l,c]=\sum_{a=1}^{k_1} \sum_{b=1}^{k_2}x[l+a,c+b]w[a,b]" />
</p>

Then the back propogation of the convolution layer uses the following expressions:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;L}{\partial&space;W[a',&space;b']}=\sum_{l=1}^{N_1}&space;\sum_{c=1}^{N_2}\frac{\partial&space;L}{\partial&space;y[l,&space;c]}x[l&plus;a',&space;c&plus;b']" title="\frac{\partial L}{\partial W[a', b']}=\sum_{l=1}^{N_1} \sum_{c=1}^{N_2}\frac{\partial L}{\partial y[l, c]}x[l+a', c+b']" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;L}{\partial&space;x[l',&space;c']}=\sum_{a=1}^{k_1}&space;\sum_{b=1}^{k_2}\frac{\partial&space;L}{\partial&space;y[l'-a,&space;c'-b]}w[a,b]" title="\frac{\partial L}{\partial x[l', c']}=\sum_{a=1}^{k_1} \sum_{b=1}^{k_2}\frac{\partial L}{\partial y[l'-a, c'-b]}w[a,b]" />
</p>

### Max pooling layer ###
The max pooling layer picks the maximum value for each region and form a smaller matrix from the original matrix. This helps reduce the spatial dimensions while keeping the important information. How max pooling back propagates is shown below.

![alt text](/images/pooling.png)

### Dropout layer ###
Dropout is a technique used to improve overfitting on neural networks, and it should also be used together with other techniques like L2 Regularization. Usually half of the neurons are randomly shut down and back propagated only in training in order to force neurons to learn more information as before. When in predicting, no dropout should be applied. How dropout works is shwon below:

![alt text](/images/dropout.jpeg)

Notice that in this implementation, inverted dropout, in which the mask is scaled by 1/p, is used to avoid multiplying p again in the back propagation. One can also scale the output in the backprop without inverted dropout. Here is a detailed explanation of [inverted dropout](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/).

## Code Usage
This is only a naive layer implementation and is only used for understanding of CNN. It cannot be used for real dataset since it would be TOO SLOW. To vectorize the implementation, refer to [im2col](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html) to find the answer.

## Reference
1. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/hw1-q6/).
2. [https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_learning.html](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_learning.html).
3. [https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/](https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/)
4. [https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/slides/L10_cnns_backprop_notes.pdf](https://www.cc.gatech.edu/classes/AY2019/cs7643_fall/slides/L10_cnns_backprop_notes.pdf)
