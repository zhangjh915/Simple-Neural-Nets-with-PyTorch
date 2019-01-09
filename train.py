# Reference: https://github.com/pytorch/examples.
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import models
from cifar10 import CIFAR10
import matplotlib.pyplot as plt


def train(args, epoch, model, train_loader, val_loader, test_loader, loss_func, opt):
    """
    Train the model for one epoch.
    Arguments:
        args: training settings
        epoch: epoch index
        model: model in one of ('softmax', 'twolayernn','convnet')
        train_loader: training data loader
        val_loader: validation data loader
        test_loader: test data loader
        loss_func: loss function, which is cross entropy loss in this repository
        opt: optimizer
    """
    model.train()
    a = list(enumerate(train_loader))
    for batch_idx, (images, targets) in enumerate(train_loader):
        if args.use_cuda:
            images, targets = images.cuda(), targets.cuda()

        loss = loss_func(model(images), targets)
        opt.zero_grad()  # clear gradients for next train
        loss.backward()  # back propagation, compute gradients
        opt.step()       # apply gradients

        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = loss_calc(args, model, val_loader, test_loader, loss_func, data_split='val')
            args.train_loss.append(loss.item())
            args.val_loss.append(val_loss)
            args.val_acc.append(val_acc)
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\t'
                  'Train Loss: {:.2f}  Validation Loss: {:.2f}  Validation Accuracy: {:.2f}%'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), val_loss, val_acc))


def loss_calc(args, model, val_loader, test_loader, loss_func, data_split, n_batches=4):
    """
    Compute loss on the validation/test set.
    Arguments:
        args: training settings
        model: model in one of ('softmax', 'twolayernn','convnet')
        val_loader: validation data loader
        test_loader: test data loader
        loss_func: loss function, which is cross entropy loss in this repository
        data_split: string of either 'val' or 'test'
        n_batches: number of batches
    Outputs:
        loss: float of loss
        acc: float of accuracy
    """
    model.eval()
    loss = 0
    num_correct = 0
    num_samples = 0
    if data_split == 'val':
        loader = val_loader
    elif data_split == 'test':
        loader = test_loader
    for batch_idx, (images, targets) in enumerate(loader):
        if args.use_cuda:
            images, targets = images.cuda(), targets.cuda()
        output = model(images)
        loss += loss_func(output, targets, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        num_correct += pred.eq(targets.data.view_as(pred)).sum().item()
        num_samples += pred.size(0)
        if n_batches and (batch_idx >= n_batches):
            break

    loss /= num_samples
    acc = num_correct / num_samples * 100.

    if data_split == 'val':
        split_name = 'Validation'
    elif data_split == 'test':
        split_name = 'Test'
    print('\n{} set: Average loss: {:.2f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        split_name, loss, num_correct, num_samples, acc))

    return loss, acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR-10 dataset')
    parser.add_argument('--batch-size', type=int, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--cifar10-dir', default='data',
                        help='directory that contains cifar-10-batches-py/ '
                        '(downloaded automatically if necessary)')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--hidden-dim', type=int,
                        help='number of hidden features/activations')
    parser.add_argument('--kernel-size', type=int,
                        help='size of convolution kernels/filters')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches between logging train status')
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--model', choices=['Softmax', 'Twolayernn', 'Convnet'],
                        help='which model to train/evaluate')
    parser.add_argument('--momentum', type=float, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=True,  # use CPU on my laptop
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='saves the current model')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay hyperparameter')
    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    # set seed
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Load CIFAR10 dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    n_classes = 10
    im_size = (3, 32, 32)
    # normalization
    cifar10_mean = [0.49131522, 0.48209435, 0.44646862]  # mean color of training images
    cifar10_std = [0.01897398, 0.03039277, 0.03872553]  # standard deviation of color across training images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std),])
    # datasets
    train_dataset = CIFAR10(args.cifar10_dir, data_split='train', transform=transform, download=True)
    val_dataset = CIFAR10(args.cifar10_dir, data_split='val', transform=transform, download=True)
    test_dataset = CIFAR10(args.cifar10_dir, data_split='test', transform=transform, download=True)
    # dataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Load model
    if args.model == 'Softmax':
        model = models.Softmax(im_size, n_classes)
    elif args.model == 'Twolayernn':
        model = models.Two_Layer_NN(im_size, args.hidden_dim, n_classes)
    elif args.model == 'Convnet':
        model = models.Conv_Net(im_size, args.hidden_dim, n_classes, args.kernel_size)
    else:
        raise Exception('Unknown model {}'.format(args.model))

    # Define loss function - Cross-Entropy Loss
    loss_func = F.cross_entropy
    if args.use_cuda:
        model.cuda()

    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Record loss and accuracy history
    args.train_loss = []
    args.val_loss = []
    args.val_acc = []

    # Train the model
    for epoch in range(1, args.epochs+1):
        train(args, epoch, model, train_loader, val_loader, test_loader, loss_func, opt)

    # Evaluate on test set
    loss_calc(args, model, val_loader, test_loader, loss_func, data_split='test')

    # Save model
    if args.save_model:
        torch.save(model, args.model + '.pt')

    # Plot loss and accuracy curves
    plt.subplot(121)
    plt.plot(args.train_loss, label='training')
    plt.plot(args.val_loss, label='validation')
    plt.title(args.model + ' Learning Curve')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(args.val_acc, label='validation')
    plt.title(args.model + ' Validation Accuracy During Training')
    plt.ylabel('Accuracy')

    plt.show()


if __name__ == '__main__':
    main()
