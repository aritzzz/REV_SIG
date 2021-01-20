import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.Context import Context


class ScaffoldPrediction(nn.Module):
	def __init__(self, out_features, ncodes):
		super(ScaffoldPrediction, self).__init__()
		#self.in_features = in_features
		self.out_features = out_features
		self.ncodes = ncodes


		self.Rec = nn.Sequential(
					nn.Linear(self.ncodes*self.out_features, 512),
					nn.ReLU(),
					# nn.Linear(512, 256),
					# nn.ReLU(),
					nn.Linear(512,1)
			)

		self.Conf = nn.Sequential(
					nn.Linear(self.ncodes*self.out_features, 512),
					nn.ReLU(),
					# nn.Linear(512, 256),
					# nn.ReLU(),
					nn.Linear(512,1)
			)


	def forward(self, rec_in, conf_in):

		rec_preds = self.Rec(rec_in)

		conf_preds = self.Conf(conf_in)
		
		return rec_preds, conf_preds



class MainPrediction(nn.Module):
	def __init__(self, in_features, out_features, ncodes):
		super(MainPrediction, self).__init__()

		self.in_features = in_features
		self.out_features = out_features
		self.ncodes = ncodes

		self.exhaustivenss_codes = Context(self.in_features, self.out_features, self.ncodes)

		self.subj_codes = Context(self.in_features, self.out_features, self.ncodes)

		self.intensity_codes = Context(self.in_features, self.out_features, self.ncodes)

		self.Exhaustivenes = nn.Sequential(
					nn.Linear(self.ncodes*self.out_features, 512),
					nn.ReLU(),
					# nn.Linear(512, 256),
					# nn.ReLU(),
					nn.Linear(512,1)
			)

		self.Subjectivity = nn.Sequential(
					nn.Linear(self.ncodes*self.out_features, 512),
					nn.ReLU(),
					# nn.Linear(512, 256),
					# nn.ReLU(),
					nn.Linear(512,1)
			)

		self.Intensity = nn.Sequential(
					nn.Linear(self.ncodes*self.out_features, 512),
					nn.ReLU(),
					# nn.Linear(512, 256),
					# nn.ReLU(),
					nn.Linear(512,1)
			)

	
	def forward(self, input, rec_codes, conf_codes):

		input = torch.cat((input, rec_codes, conf_codes), 1)


		print("In main Prediction: input shape {}, rec_codes shape {}, conf_codes shape".format(input.shape, rec_codes.shape, conf_codes.shape))

		exhaus_codes = self.exhaustivenss_codes(input)
		subj_codes = self.subj_codes(input)
		intense_codes = self.intensity_codes(input)

		# exhaus_codes = torch.cat((exhaus_codes, rec_codes, conf_codes), 1)
		# subj_codes = torch.cat((subj_codes, rec_codes, conf_codes), 1)
		# intense_codes = torch.cat((intense_codes, rec_codes, conf_codes), 1)

		#print(exhaus_codes.shape, subj_codes.shape, intense_codes.shape)
		ex_preds = self.Exhaustivenes(exhaus_codes.view(input.shape[0], -1))
		subj_preds = self.Subjectivity(subj_codes.view(input.shape[0], -1))
		intensity_preds = self.Intensity(intense_codes.view(input.shape[0], -1))

		return ex_preds, subj_preds, intensity_preds