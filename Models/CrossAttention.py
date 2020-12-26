import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class CrossAttention(nn.Module):
	def __init__(self, in_features, out_features):
		super(CrossAttention, self).__init__()

		self.in_features = in_features
		self.out_features = out_features

		self.linear1 = nn.Linear(self.in_features, self.out_features, bias=True)
		self.relu = nn.ReLU()


	def forward(self, paper, review):
		"""
			paper shape (bsz, dim, seq_len)
			review shape (bsz, dim, seq_len)
		"""
		paper = paper.transpose(1,2)
		review = review.transpose(1,2)
		dim = review.shape[1]
		paper_linear = self.relu(self.linear1(paper))
		review_linear = self.relu(self.linear1(review))
		Affinity = torch.bmm(review_linear, paper_linear.transpose(1,2))/(np.sqrt(dim)) #shape = (_, R, P)
		C = F.softmax(Affinity, dim=1)  #normalized across columns
		Rp = torch.sum(review.unsqueeze(-1)*C.unsqueeze(2), dim=-1)
		Pr = torch.bmm(C, paper)
		Rc = torch.cat((review, Rp, Pr), dim=1)

		#Alignment = torch.bmm(F.softmax(Affinity, dim=1), paper.transpose(1,2))
		return Rp, Pr, Rc

	def init_weights(self):
		nn.init.xavier_normal_(self.linear1.weight)