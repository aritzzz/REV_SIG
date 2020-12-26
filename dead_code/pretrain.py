import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformer import Encoder, PositionalEncoding
from data_prepare import *


""" Decoder Definition """
class Decoder(nn.Module):
    def __init__(self, ntoken_trg, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(Decoder, self).__init__()
        self.pos_decoder = PositionalEncoding(ninp, dropout)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.decoder_emb = nn.Embedding(ntoken_trg, ninp)
        self.pred_decoder = nn.Linear(ninp, ntoken_trg)
        self.ninp = ninp
        self.init_weights()

    def forward(self, enc_output, trg):
        trg_device = trg.device
        trg_mask = Decoder._generate_square_subsequent_mask(len(trg)).to(trg_device)
        trg_key_padding_mask = Decoder._generate_padding_mask(trg).to(trg_device)
        trg = self.decoder_emb(trg) * math.sqrt(self.ninp)
        trg = self.pos_decoder(trg)
        dec_output = self.transformer_decoder(trg, enc_output, tgt_mask=trg_mask,\
                                            tgt_key_padding_mask=trg_key_padding_mask)
        preds = self.pred_decoder(dec_output)
        return preds

    @staticmethod
    def _generate_padding_mask(tsr):
        # padding index is 0
        msk = torch.tensor((tsr.data.cpu().numpy() == 0).astype(int), dtype = torch.bool)
        msk = msk.permute(1,0)
        return msk

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_emb.weight.data.uniform_(-initrange, initrange)
        self.pred_decoder.bias.data.zero_()
        self.pred_decoder.weight.data.uniform_(-initrange, initrange)


if __name__ == "__main__":
    main_task_loader, scaffold_task_loader, test_loader, vocab = getLoaders(['./Data/SignData/'], ['./Data/2018/', './Data/2019/'], batch_size=8,split=True)
    ntokens = len(vocab)
    print("Vocab size: {}".format(ntokens))
    encoder = Encoder(ntokens,256,2,256,2)
    decoder = Decoder(ntokens,256,2,256,2)
    decoder.decoder_emb.weight.data.copy_(encoder.encoder_emb.weight.data)
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters())  #, lr=0.0001)#, lr=0.002, weight_decay=0.5, momentum=0.99)
    decoder_optimizer = torch.optim.Adam(decoder.parameters())

    epochs = 50
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0.0
        encoder.train()
        decoder.train()
        for i, data in enumerate(scaffold_task_loader,0):
            # print(data['paper'])
            paper = data['paper'].transpose(0,1)
            review = data['review'].transpose(0,1)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            _, (enc_paper, enc_review) = encoder(None,(paper,review))

            # print(enc_paper.shape, enc_review.shape)
            out_paper = decoder(enc_paper, paper[:-1,:])
            out_review = decoder(enc_review, review[:-1,:])
            # print(out_paper.view(-1,ntokens).shape)
            # print(paper[1:,:].reshape(-1).shape)
            loss_paper = criterion(out_paper.view(-1,ntokens), paper[1:,:].reshape(-1))
            loss_review = criterion(out_review.view(-1,ntokens), review[1:,:].reshape(-1))
            print("Iteration {} Loss Paper {} Loss Review {}".format(i, loss_paper.item(), loss_review.item()))
            loss = loss_paper + loss_review
            loss.backward()
            epoch_loss += loss.item()
            decoder_optimizer.step()
            encoder_optimizer.step()
            print("Iteration {} Loss {}".format(i, loss.item()))

        eval_epoch_loss = 0.0
        with torch.no_grad():
            for i, data_ in enumerate(test_loader,0):
                paper = data_['paper'].transpose(0,1)
                review = data_['review'].transpose(0,1)
                _, (enc_paper, enc_review) = encoder(None,(paper,review))

                out_paper = decoder(enc_paper, paper[:-1,:])
                out_review = decoder(enc_review, review[:-1,:])

                loss_paper = criterion(out_paper.view(-1,ntokens), paper[1:,:].reshape(-1))
                loss_review = criterion(out_review.view(-1,ntokens), review[1:,:].reshape(-1))
                print("Eval : Iteration {} Loss Paper {} Loss Review {}".format(i, loss_paper.item(), loss_review.item()))
                loss = loss_paper + loss_review
                eval_epoch_loss += loss.item()
                print("Eval : Iteration {} Loss {}".format(i, loss.item()))

        if (eval_epoch_loss/(len(test_loader))) < best_val_loss:
            best_val_loss = eval_epoch_loss/(len(test_loader))
            torch.save({'encoder': encoder.state_dict(),\
                        'decoder': decoder.state_dict()}, 'checkpoint2.pt')

        print("Train Epoch: {} Loss {}".format(epoch, epoch_loss/len(scaffold_task_loader)))
        print("Eval Epoch: {} Loss {}".format(epoch, eval_epoch_loss/len(test_loader)))

        #     break
        # break





