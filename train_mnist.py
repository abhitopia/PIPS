# Reference: https://github.com/leequant761/Gumbel-VAE/

# Main Reference : https://github.com/daandouwe/concrete-vae
import argparse
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Decoder(nn.Module):
    def __init__(self, latent_num=30, latent_dim=10, hidden=200, output=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_num*latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.act_fn = nn.Tanh()

    def forward(self, z):
        h = self.fc1(z)
        h = self.act_fn(h)
        return self.fc2(h)


class Encoder(nn.Module):
    def __init__(self, input=784, hidden=200, latent_num=30, latent_dim=10):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, latent_num * latent_dim)
        self.act_fn = nn.Tanh()
        self.latent_num = latent_num
        self.latent_dim = latent_dim

    def forward(self, x):
        h = self.fc1(x)
        h = self.act_fn(h)
        return self.fc2(h).reshape(-1, self.latent_num, self.latent_dim)

class Model(nn.Module):
    def __init__(self, temp, latent_num, latent_dim):
        super(Model, self).__init__()
        if type(temp) != torch.Tensor:
            temp = torch.tensor(temp)
        self.__temp = temp
        self.latent_num = latent_num
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_num=latent_num, latent_dim=latent_dim)
        self.decoder = Decoder(latent_num=latent_num, latent_dim=latent_dim)
        if 'ExpTDModel' in  str(self.__class__):
            self.prior = ExpRelaxedCategorical(temp, probs=torch.ones(latent_dim))
        else:
            self.prior = dist.RelaxedOneHotCategorical(temp, probs=torch.ones(latent_dim))
        self.initialize()

        self.softmax = nn.Softmax(dim=-1)

    @property
    def temp(self):
        return self.__temp

    @temp.setter
    def temp(self, value):
        self.__temp = value
        if 'ExpTDModel' in  str(self.__class__):
            self.prior = ExpRelaxedCategorical(value, probs=torch.ones(self.latent_dim))
        else:
            self.prior = dist.RelaxedOneHotCategorical(value, probs=torch.ones(self.latent_dim))

    def initialize(self):
        for param in self.parameters():
            if len(param.shape) > 2:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        log_alpha = self.encode(x)
        z, v_dist = self.sample(log_alpha, self.temp)
        if 'ExpTDModel' in  str(self.__class__):
            x_recon = self.decode(z.exp().view(-1, self.latent_num*self.latent_dim))
        else:
            x_recon = self.decode(z.view(-1, self.latent_num*self.latent_dim))
        return z, x_recon, v_dist

    def approximate_loss(self, x, x_recon, v_dist, eps=1e-3):
        """ KL-divergence follows Eric Jang's trick
        """
        log_alpha = v_dist.logits
        bce = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        num_class = torch.tensor(self.latent_dim).float()
        probs = torch.softmax(log_alpha, dim=-1) # alpha_i / alpha_sum
        kl = torch.sum(probs * (num_class * (probs + eps)).log(), dim=-1).sum()
        return bce, kl

    def loss(self, x, x_recon, z, v_dist):
        """ Monte-Carlo estimate KL-divergence
        """
        bce = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        n_batch = x.shape[0]
        prior = self.prior.expand(torch.Size([n_batch, self.latent_num]))
        kl = (v_dist.log_prob(z) - prior.log_prob(z)).sum()
        return bce, kl

    def sample(self, log_alpha, temp):
        raise ValueError("Not Implemented")

class TDModel(Model):
    def sample(self, log_alpha, temp):
        v_dist = dist.RelaxedOneHotCategorical(temp, logits=log_alpha)
        concrete = v_dist.rsample()
        return concrete, v_dist

class ExpTDModel(Model):
    def sample(self, log_alpha, temp):
        v_dist = ExpRelaxedCategorical(temp, logits=log_alpha)
        log_concrete = v_dist.rsample()
        return log_concrete, v_dist
    


def compute_loss(x, x_recon, z, v_dist, model, args):
    if args.kld == 'eric':
        bce, kl = model.approximate_loss(x, x_recon, v_dist)
    else:
        bce, kl = model.loss(x, x_recon, z, v_dist)

    # Normalize the losses by batch size and dimensions
    batch_size = x.size(0)
    input_size = 784  # MNIST images are 28x28 = 784 pixels
    
    # Normalize BCE by both batch size and number of pixels
    # bce = bce / (batch_size * input_size)
    bce = bce / batch_size
    
    # Normalize KL by batch size
    kl = kl / batch_size
    elbo = -bce - kl
    loss = -elbo
    return loss, bce, kl

def train(epoch, model, train_loader, optimizer, device, args):
    model = model.train()

    train_loss = 0.
    train_bce = 0.
    train_kld = 0
    for i, (x, _) in enumerate(train_loader, 1):
        x = x.to(device)
        x = x.view(x.size(0), -1)

        n_updates = epoch * len(train_loader) + i

        # scheduler for temperature
        if n_updates % args.temp_interval == 0:
            temp = max(torch.tensor(args.init_temp) * np.exp(-n_updates*args.temp_anneal), torch.tensor(args.min_temp))
            model.temp = temp
        
        # compute & optimize the loss function
        z, x_recon, v_dist = model(x)
        loss, bce, kl = compute_loss(x, x_recon, z, v_dist, model, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"Epoch: {epoch}\tStep: {i}\tKL: {kl}\t BCE: {bce} \tT: {model.temp}")
    
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kl.item()


    avg_loss = train_loss / len(train_loader)
    avg_bce = train_bce / len(train_loader)
    avg_kld = train_kld / len(train_loader)
    # report results
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} Avg KLD: {avg_kld:.4f} Avg BCE: {avg_bce:.4f}')

def test(epoch, model, test_loader, device, args):
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x = x.view(x.size(0), -1)
            # compute the loss function
            z, x_recon, v_dist = model(x)
            loss, bce, kl = compute_loss(x, x_recon, z, v_dist, model, args)
            test_loss += loss

            # save reconstructed figure
            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n].reshape(-1, 1, 28, 28),
                                        x_recon.view(args.batch_size, 1, 28, 28)[:n]])
                if 'results' not in os.listdir():
                    os.mkdir('results')
                save_image(comparison.cpu(),
                            'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    # report results
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 4} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data, train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.sampling == 'TDModel':
        Model = TDModel
    elif args.sampling == 'ExpTDModel':
        Model = ExpTDModel
    else:
        raise ValueError(f"Invalid sampling method: {args.sampling}")

    model = Model(args.init_temp, args.latent_num, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, device, args)
        test(epoch, model, test_loader, device, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='.data/',
                        help='where is you mnist?')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--latent-num', type=int, default=20, metavar='N',
                        help='the number of latent variables')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N',
                        help='the dimension for each latent variables')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--init-temp', type=float, default=1.0)
    parser.add_argument('--temp-anneal', type=float, default=0.00009)
    parser.add_argument('--temp-interval', type=float, default=300)
    parser.add_argument('--min-temp', type=float, default=0.1)

    parser.add_argument('--sampling', type=str, default='TDModel',
                        help='example: TDModel utilizes torch.distributions.relaxed, ExpTDModel stabilizes loss function')
    parser.add_argument('--kld', type=str, default='eric',
                        help='example: eric, madisson')

    args = parser.parse_args()
    main(args)