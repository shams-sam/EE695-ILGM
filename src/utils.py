from collections import OrderedDict
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def draw_weights(model_mu, model_var, device):
    weights = OrderedDict()
    for name in model_mu.keys():
        mu = model_mu[name]
        var = model_var[name]
        weights[name] = torch.normal(mean=0.0, std=1.0, size=mu.size())
        weights[name] = (weights[name]*torch.pow(var, 0.5) + mu).to(device)
    return weights


def init_model(model, mean=0.0, std=0.0, init='normal'):
    if init == 'normal':
        initializer = partial(torch.nn.init.normal_, std=0.01)
    elif init == 'uniform':
        initializer = partial(torch.nn.init.uniform_, b=0.001)
    elif init == 'orthogonal':
        initializer = torch.nn.init.orthogonal_

    for param in model.parameters():
        initializer(param)


def num_layers(model):
    count = 0
    for _ in model.parameters():
        count += 1

    return count


def plot_density(model, cols=4, bins=10, use_sns=True):
    L = num_layers(model)
    rows = L//cols
    if rows*cols < L:
        rows += 1

    fig = plt.figure(figsize=(5*cols, 4*rows))
    plt_count = 1
    for name, param in model.named_parameters():
        ax = fig.add_subplot(rows, cols, plt_count)
        if use_sns:
            ax = sns.histplot(
                to_numpy(param), bins=bins, stat='count', alpha=1.0,
                kde=True, edgecolor='white', color='b', linewidth=1.5,
                line_kws=dict(
                    color='black', alpha=1.0, linewidth=5, label='KDE'
                )
            )
            ax.get_lines()[0].set_color('red')
        else:
            ax.hist(
                to_numpy(param), bins=bins, rwidth=0.9, color='b')
        ax.grid('on')
        ax.set_title('{}: {}'.format(name, [*param.size()]))

        plt_count += 1

    plt.show()


def plot_training(epoch, acc, loss):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(epoch, loss, 'r')
    ax2.plot(epoch, acc, 'b')

    ax1.set_xlabel('epochs')
    ax2.set_xlabel('epochs')
    ax1.set_ylabel('cross entropy')
    ax2.set_ylabel('accuracy')
    ax1.grid('on')
    ax2.grid('on')

    plt.show()


def sns_density(model, cols=4, bins=10):
    L = num_layers(model)
    rows = L//cols
    if rows*cols < L:
        rows += 1

    fig = plt.figure(figsize=(5*cols, 4*rows))
    plt_count = 1
    for name, param in model.named_parameters():
        ax = fig.add_subplot(rows, cols, plt_count)
        ax.hist(
            to_numpy(param), bins=bins, hist_type='step', color='b')
        ax.grid('on')
        ax.set_title('{}: {}'.format(name, [*param.size()]))

        plt_count += 1

    plt.show()


def to_numpy(tensor, flatten=True):
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.device != 'cpu':
        tensor = tensor.cpu()
    if flatten:
        tensor = tensor.flatten()

    return tensor.numpy()


def validate(model, loader, device):
    if not device:
        device = torch.device('cpu')

    correct = 0.0
    total = 0
    for Y, X in loader:
        Y, X = Y.to(device), X.to(device)
        X_ = model(Y).argmax(axis=1, keepdim=True)
        correct += X_.eq(X.view_as(X_)).sum().item()
        total += len(X)

    return correct/total
