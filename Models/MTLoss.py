import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MTLoss(nn.Module):
	def __init__(self):
		super(MTLoss, self).__init__()

		self.log_var_a = nn.Parameter(torch.zeros((1,)))
		self.log_var_b = nn.Parameter(torch.zeros((1,)))
		self.log_var_c = nn.Parameter(torch.zeros((1,)))
		self.log_var_d = nn.Parameter(torch.zeros((1,)))
		self.log_var_e = nn.Parameter(torch.zeros((1,)))

		self.log_vars = [self.log_var_a, self.log_var_b, self.log_var_c, self.log_var_d, self.log_var_e]
		
	def forward(self, predictions, targets):
		loss = 0
		for i in range(len(predictions)):
			precision = torch.exp(-self.log_vars[i])
			diff = (predictions[i] - targets[i])**2
			temp = precision*diff + self.log_vars[i]
			loss += torch.sum(precision*diff + self.log_vars[i], -1)
		return torch.mean(loss)



if __name__ == "__main__":
	loss_fn = MTLoss()
	print(loss_fn)
	for name, param in loss_fn.named_parameters():
		print(name, param)