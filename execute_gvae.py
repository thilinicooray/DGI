import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import gvae, LogReg
from utils import process

import torch.nn.functional as F

from utils_vae import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence, \
    accumulate_group_rep, expand_group_rep



class GcnInfomax(nn.Module):
    def __init__(self, ft_size, hid_units, nonlinearity):
        super(GcnInfomax, self).__init__()

        self.encoder = gvae.Encoder(ft_size, hid_units, nonlinearity)
        self.decoder = gvae.Decoder(hid_units, hid_units, ft_size)


        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self , seq1, org_adj, adj, sparse, msk, samp_bias1, samp_bias2):

        node_mu, node_logvar = self.encoder(seq1, None, adj, sparse, msk, samp_bias1, samp_bias2)


        node_kl_divergence_loss = -0.5 * torch.mean(torch.sum(
            1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))


        node_kl_divergence_loss = node_kl_divergence_loss


        node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)


        reconstructed_node = self.decoder(node_latent_embeddings)

        reconstruction_error = self.recon_loss(reconstructed_node, org_adj)


        loss =  node_kl_divergence_loss + reconstruction_error

        loss.backward()


        return  loss

    def recon_loss(self, recon_node, adj):

        b_xent = nn.BCEWithLogitsLoss()

        recon = recon_node[0]

        adj = adj[0]
        recon_adj = torch.sigmoid(torch.mm(recon, recon.t()))

        pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        cost = norm * b_xent(recon_adj, adj, pos_weight=pos_weight)

        return cost

    def get_embeddings(self, seq, adj, sparse, msk):

        node_mu, node_logvar = self.encoder.embed(seq, adj, sparse, msk)

        node_latent_embeddings = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

        return node_latent_embeddings



dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
#if not sparse:
adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = GcnInfomax(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
        adj = adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()


    loss = model(features, adj, sp_adj if sparse else adj, sparse, None, None, None)

    print('Loss:', loss.item())

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))

embeds = model.get_embeddings(features, sp_adj if sparse else adj, sparse, None)
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

tot = torch.zeros(1)
tot = tot.cuda()

accs = []

for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    best_acc = best_acc.cuda()
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print('Average accuracy:', tot / 50)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())

