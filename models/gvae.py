import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator
from collections import OrderedDict
from torch.nn import Sequential, Linear, ReLU, Tanh, Sigmoid, PReLU


class Encoder(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(Encoder, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.mu = GCN(n_h, n_h*2, activation)
        self.logvar = GCN(n_h, n_h*2, activation)


    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        mu = self.mu(h_1, adj, sparse)
        log_var = self.logvar(h_1, adj, sparse)


        return mu, log_var

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        mu = self.mu(h_1, adj, sparse)
        log_var = self.logvar(h_1, adj, sparse)


        return mu, log_var

class Decoder(torch.nn.Module):
    def __init__(self, node_dim, class_dim, feat_size):
        super(Decoder, self).__init__()

        self.linear_model = torch.nn.Sequential(OrderedDict([
            ('linear_1', torch.nn.Linear(in_features=node_dim + class_dim, out_features=node_dim, bias=True)),
            ('relu_1', ReLU()),

            ('linear_2', torch.nn.Linear(in_features=node_dim, out_features=feat_size, bias=True)),
            ('relu_final', Tanh())
        ]))

    def forward(self, x):
        #x = torch.cat((node_latent_space, class_latent_space), dim=1)

        #x = torch.softmax(self.linear_model(x), dim=-1)
        x = self.linear_model(x)

        return x

