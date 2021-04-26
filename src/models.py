import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))

        return self.fc2(x)


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=[], output_size=10):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.layer_sizes = [input_size] + hidden_size + [output_size]
        layers = []
        for i in range(1, len(self.layer_sizes)-1):
            layers.append(
                nn.Linear(
                    self.layer_sizes[i-1],
                    self.layer_sizes[i]
                )
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(
                self.layer_sizes[-2],
                self.layer_sizes[-1]
            )
        )

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x.view(-1, self.input_size))
