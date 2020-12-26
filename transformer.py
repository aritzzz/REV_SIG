import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from data_prepare_copy import get_stat

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)






# class Encoder(nn.Module):
#     def __init__(self, ntokens, ninp, nhead, nhid, nlayers, dropout=0.5):
#         super(Encoder, self).__init__()
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.encoder_emb = nn.Embedding(ntokens, ninp)
#         self.ninp = ninp
#         self.init_weights()

#     def forward(self, src):
#         src_device = src.device
#         src_key_padding_mask = Encoder._generate_padding_mask(src).to(src_device)
#         src = self.encoder_emb(src) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         enc_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
#         return enc_output

#     @staticmethod
#     def _generate_padding_mask(tsr):
#         # padding index is 0
#         msk = torch.tensor((tsr.data.cpu().numpy() == 0).astype(int), dtype = torch.bool)
#         msk = msk.permute(1,0)
#         return msk

#     def init_weights(self):
#         initrange = 0.1
#         self.encoder_emb.weight.data.uniform_(-initrange, initrange)

class reviewContext(nn.Module):
    def __init__(self, ncodes):
        super(reviewContext, self).__init__()
        self.ncodes = ncodes
        self.codes = nn.Linear(256,self.ncodes, bias=False)
        self.init_weights()


    def forward(self, rev_repr): #rev_repr shape = (bsz, 512, 100)
        print("Inside Coder: {}".format(rev_repr.shape))
        # print(self.codes(rev_repr).shape)
        wts =  F.softmax(self.codes(rev_repr), dim=1) #shape = (bsz, 512, 3)
        # print(wts.shape)
        wts = wts.unsqueeze(-1)
        # print(wts.shape)
        rev_repr = rev_repr.unsqueeze(2)
        # # print(wts.shape)
        # print(rev_repr.shape)
        temp = wts*rev_repr
        # print(temp.shape)
        contexts = torch.sum(temp, dim=1)
        # contexts = None
        print(contexts.shape)
        return contexts

    def init_weights(self):
        nn.init.xavier_normal_(self.codes.weight)

class MainTask1(nn.Module):
    def __init__(self):
        super(MainTask1, self).__init__()
        self.main = nn.Sequential(
                        nn.Linear(512,128),
                        nn.ReLU(),
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Linear(64,32),
                        nn.Linear(32,1)
                        )

    def forward(self, output):
        return self.main(output)

class MainTask2(nn.Module):
    def __init__(self):
        super(MainTask2, self).__init__()
        self.main = nn.Sequential(
                        nn.Linear(512,128),
                        nn.ReLU(),
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Linear(64,32),
                        nn.Linear(32,1)
                        )

    def forward(self, output):
        return self.main(output)

class MainTask3(nn.Module):
    def __init__(self):
        super(MainTask3, self).__init__()
        self.main = nn.Sequential(
                        nn.Linear(512,128),
                        nn.ReLU(),
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Linear(64,32),
                        nn.Linear(32,1)
                        )

    def forward(self, output):
        return self.main(output)

class RecommendationTask(nn.Module):
    def __init__(self):
        super(RecommendationTask, self).__init__()
        self.recommendation = nn.Sequential(
                        # nn.Linear(1024,512),
                        # nn.ReLU(),
                        # nn.Linear(512,256),
                        # nn.ReLU(),
                        nn.Linear(8192,128),
                        nn.ReLU(),
                        # nn.Linear(384,128),
                        # nn.ReLU(),
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Linear(64,1),
                        # nn.ReLU(),
                        # nn.Linear(32,1)
                        # nn.ReLU(),
                        # nn.Linear(16,1)
                        )

    def forward(self, output):
        return self.recommendation(output)

class ConfidenceTask(nn.Module):
    def __init__(self):
        super(ConfidenceTask, self).__init__()
        self.confidence = nn.Sequential(
                        nn.Linear(512,256),
                        nn.ReLU(),
                        nn.Linear(256,128),
                        nn.ReLU(),
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Linear(64,32),
                        nn.Linear(32,1)
                        )

    def forward(self, output):
        return self.confidence(output)



class Encoder(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.2):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # self.rec = RecommendationTask()
        self.init_weights()


    def forward_once(self, data):
        paper,review = data
        # print("Inside forward_once, paper:{} review: {}".format(paper.shape, review.shape))
        paper_key_padding_mask = Encoder._generate_padding_mask(paper)#.to(src_device)
        paper = self.encoder_emb(paper) * math.sqrt(self.ninp)
        paper = self.pos_encoder(paper)
        paper_enc_output = self.transformer_encoder(paper, src_key_padding_mask=paper_key_padding_mask)

        review_key_padding_mask = Encoder._generate_padding_mask(review)#.to(src_device)
        review = self.encoder_emb(review) * math.sqrt(self.ninp)
        review = self.pos_encoder(review)
        review_enc_output = self.transformer_encoder(review, src_key_padding_mask=review_key_padding_mask)

        return paper_enc_output, review_enc_output



    def forward(self, main_data, scaffold_data):

        # stator = get_stat(``)
        # print([i for i in enumerate(self.rec)])


        if main_data != None:
            out_main = self.forward_once(main_data)
        else:
            out_main = None
        out_scaffold = self.forward_once(scaffold_data)
        # scaffold_review_repr = torch.mean(out_scaffold[1].transpose(0,1), dim=1)
        # print(stator.act_means)
        # stator.plot_stat(step)
        # pred_rec = self.rec(scaffold_review_repr)
        return out_main, out_scaffold


    @staticmethod
    def _generate_padding_mask(tsr):
        # padding index is 0
        msk = torch.tensor((tsr.data.cpu().numpy() == 0).astype(int), dtype = torch.bool)
        msk = msk.permute(1,0)
        return msk

    def init_weights(self):
        initrange = 0.1
        self.encoder_emb.weight.data.uniform_(-initrange, initrange)
