import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class GCN_org(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN_org, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class GCN(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 in_ft, out_ft, act, bias=True):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCN, self).__init__()



        self.ingc_g = GCN_org(out_ft, out_ft, act)
        self.ingc = GCN_org(in_ft, out_ft, act)
        self.midlayer = nn.ModuleList()
        for i in range(1):
            gcb = GCN_org(out_ft, out_ft, act)
            self.midlayer.append(gcb)


    def reset_parameters(self):
        pass


    def get_mask(self, adj):
        fully_connected = torch.ones_like(adj).cuda()
        mask = fully_connected.masked_fill(adj > 0, 0)
        return mask

    def forward(self, fea, adj, sparse=False):

        #print(fea.size(), adj.size())
        fea = torch.squeeze(fea)
        adj = torch.squeeze(adj)

        flag_adj = adj.masked_fill(adj > 0, 1)

        x_enc = self.ingc(fea, adj, sparse)

        x = F.dropout(x_enc, 0.8, training=self.training)


        val = self.ingc_g(x, self.get_mask(adj), False)
        #val_in = val + x
        val_org = val

        mask = flag_adj
        orgx = x
        tot = x


        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            current_layer_adj = torch.mm(mask, flag_adj)
            mask = mask + current_layer_adj

            midgc = self.midlayer[i]
            x = midgc(x, adj, sparse)
            x = F.dropout(x, self.dropout, training=self.training)

            new_val = midgc(x, self.get_mask(mask))
            val = val + F.dropout(new_val, 0.8, training=self.training)

            mfb_sign_sqrt = torch.sqrt(F.relu(val)) - torch.sqrt(F.relu(-(val)))

            val = F.normalize(mfb_sign_sqrt)

        return x, val

    def rank_loss(self, org_feat, local_rep, non_local_rep, non_local_org):
        non_loc_sim = torch.bmm(org_feat.view(org_feat.size(0), 1, org_feat.size(1))
                                , non_local_rep.view(non_local_rep.size(0), non_local_rep.size(1), 1))

        non_loc_sim_org = torch.bmm(non_local_org.view(non_local_org.size(0), 1, non_local_org.size(1))
                                    , org_feat.view(org_feat.size(0), org_feat.size(1), 1))

        loc_sim = torch.bmm(non_local_org.view(non_local_org.size(0), 1, non_local_org.size(1))
                            , local_rep.view(local_rep.size(0), local_rep.size(1), 1))

        margin = torch.bmm(non_local_rep.view(non_local_rep.size(0), 1, non_local_rep.size(1))
                           , local_rep.view(local_rep.size(0), local_rep.size(1), 1))


        marginal_rank_loss = torch.mean(torch.max(torch.zeros(org_feat.size(0)).cuda(), margin.squeeze()  - non_loc_sim.squeeze() ),0) + \
                             torch.mean(torch.max(torch.zeros(org_feat.size(0)).cuda(), non_loc_sim_org.squeeze()  - loc_sim.squeeze() ),0)

        return 0* marginal_rank_loss

