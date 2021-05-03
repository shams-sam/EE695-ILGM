import torch
import torch.distributions as distrib
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


class Decoder(nn.Module):
    def __init__(self, n_in, n_latent, n_hidden, n_classes):
        super(Decoder, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_in*n_classes)
        )

    def forward(self, x):
        return self.pipe(x)


class Encoder(nn.Module):
    def __init__(self, n_in, n_hidden):
        super(Encoder, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
        )

    def forward(self, x):
        return self.pipe(x)


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



# Reference:
# https://github.com/acids-ircam/pytorch_flows/blob/master/flows_03.ipynb
class VAE(nn.Module):

    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims
        self.mu = nn.Linear(encoder_dims, latent_dims)
        self.sigma = nn.Sequential(
            nn.Linear(encoder_dims, latent_dims),
            nn.Softplus(),
            nn.Hardtanh(min_val=1e-4, max_val=5.))
        self.apply(self.init_parameters)

    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(x, z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, kl_div

    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Retrieve mean and var
        mu, sigma = z_params
        # Re-parametrize
        q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        z = (sigma * q.sample((n_batch, ))) + mu
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        kl_div = kl_div / n_batch
        return z, kl_div
