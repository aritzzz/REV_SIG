import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Context(nn.Module):
    def __init__(self, in_features, out_features, ncodes):
        super(Context, self).__init__()
        self.ncodes = ncodes
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.codes = nn.Linear(self.out_features,self.ncodes, bias=False)
        self.act = nn.ReLU()
        #self.init_weights()


    def forward(self, review): #review shape = (bsz, dim, seq_len)
        #print("Inside Coder: {}".format(rev_repr.shape))
        #print(self.codes(rev_repr).shape)
        #review = review.transpose(1,2)
        #print(review.shape)
        rev_repr = self.act((self.linear(review)))
        #print(rev_repr.shape)

        wts =  F.softmax(self.codes(rev_repr), dim=1) #shape = (bsz, 512, 3)
        #print(wts.shape)
        wts = wts.unsqueeze(-1)
        # print(wts.shape)
        rev_repr = rev_repr.unsqueeze(2)
        # # print(wts.shape)
        # print(rev_repr.shape)
        temp = wts*rev_repr
        # print(temp.shape)
        contexts = torch.sum(temp, dim=1)
        # contexts = None
        # print(contexts.shape)
        return contexts

    def init_weights(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_normal_(self.codes.weight)