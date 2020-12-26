import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from Models.Context import Context
from Models.CrossAttention import CrossAttention
#from Models.Prediction import ScaffoldPrediction


class Pipeline(nn.Module):
	def __init__(self, args):
		super(Pipeline, self).__init__()

		self.args = args
		self._createModel()

	def _createModel(self):
		self.cross_attention = CrossAttention(self.args.dim, self.args.upscale_dim)
		coders = []
		dim = self.args.dim
		for i, ncode in enumerate(map(int,self.args.codes.split(','))):
			coders.append(('coder'+str(i), Context(dim, self.args.upscale_dim, ncode)))
			dim = self.args.upscale_dim

		self.contextor = nn.Sequential(OrderedDict(coders))


		self.rec_codes = Context(self.args.upscale_dim, self.args.upscale_dim, 8)

		self.conf_codes = Context(self.args.upscale_dim, self.args.upscale_dim, 8)

		#self.scaffold_predictor = ScaffoldPrediction(self.args.upscale_dim, 8)

	def forward(self, paper, review):
		Rp, Pr, Rc = self.cross_attention(paper, review)

		out_reviews = self.contextor(review.transpose(1,2))

		out_Rp = self.contextor(Rp)
		out_Pr = self.contextor(Pr)
		out_Rc = self.contextor(Rc)
		
		out = torch.cat((out_reviews, out_Rp, out_Pr, out_Rc), dim=1)

		rec_codes = self.rec_codes(out)
		conf_codes = self.conf_codes(out)

		# predictions = self.scaffold_predictor(rec_codes.view(out.shape[0], -1), conf_codes.view(out.shape[0], -1))
		return out, rec_codes, conf_codes





