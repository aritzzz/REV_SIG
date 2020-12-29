# predict the significance scores of a test review
import sys
import os
import json
import re
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from loadData import preprocess, Embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import argparse
import numpy as np
from Models import Pipeline, MTLoss, Prediction #CrossAttention, Context
from types import SimpleNamespace



class REVSIGModel(object):
	def __init__(self, main_model, main_task_predictor, scaffold_task_predictor):
		self.model = main_model
		self.main_task_predictor = main_task_predictor
		self.scaffold_task_predictor = scaffold_task_predictor

	@classmethod
	def Initialize(cls, checkpoint_path):
		device = 'cpu'
		checkpoint = torch.load(checkpoint_path, map_location=device)
		args = SimpleNamespace(dim = checkpoint['dim'], upscale_dim = checkpoint['upscale_dim'], codes=checkpoint['codes']) 
		model = Pipeline.Pipeline(args).to(device)
		model.load_state_dict(checkpoint['model_state_dict'])
		main_task_predictor = Prediction.MainPrediction(args.upscale_dim, args.upscale_dim, 32).to(device)
		main_task_predictor.load_state_dict(checkpoint['main_state_dict'])
		scaffold_task_predictor = Prediction.ScaffoldPrediction(args.upscale_dim, 8).to(device)
		scaffold_task_predictor.load_state_dict(checkpoint['scaffold_state_dict'])
		return cls(model, main_task_predictor, scaffold_task_predictor)

	def predict(self, paper_embed, review_embed):
		paper, review = torch.from_numpy(paper_embed).unsqueeze(0), torch.from_numpy(review_embed)
		paper, review = paper.transpose(1,2).float().to(self.device),review.transpose(1,2).float().to(self.device)
		out, rec_codes, conf_codes = self.model(paper, review)
		rec_preds, conf_preds = self.scaffold_task_predictor(rec_codes.view(out.shape[0], -1), conf_codes.view(out.shape[0], -1))
		ex_preds, subj_preds, intensity_preds = self.main_task_predictor(out, rec_codes, conf_codes)

		return {'Recommendation': np.round(rec_preds.item(), 3), 'Confidence': np.round(conf_preds.item(), 3),\
			'Exhaustive': np.round(ex_preds.item(), 3), 'Aspectual Score': np.round(subj_preds.item(), 3), 'Intensity': np.round(intensity_preds.item(), 3)}

	@property
	def device(self):
		return next(self.model.parameters()).device
		
	



class Prepare(object):
	def __init__(self, paper, reviews, recs, confs):
		self.paper = paper
		self.reviews = reviews
		self.recs = recs
		self.confs = confs

	@classmethod
	def fromJson(cls, path, ID):
		with open(os.path.join(path, str(ID)+'.pdf.json'), 'r') as f:
			paper = json.load(f)
		paper_content = Prepare.get_paper_content(paper['metadata'])
		with open(os.path.join(path, str(ID)+'.json'), 'r') as f:
			reviews = json.load(f)['reviews']

		review_contents = []
		recommendations = []
		confidences = []
		for review in reviews:
			review_contents.append(review.get('comments', ''))
			recommendations.append(review.get('RECOMMENDATION', None))
			confidences.append(review.get('CONFIDENCE', None))
		
		return cls(paper_content, review_contents, recommendations, confidences)
	
	def Embed(self):
		ret = []
		for i,review in enumerate(self.reviews):
				ret.append((Embeddings(sent_tokenize(preprocess(self.paper)), [sent_tokenize(preprocess(review))]), self.recs[i], self.confs[i], preprocess(review)))
		return ret

	@staticmethod
	def get_paper_content(paper_dict):
		if paper_dict.get('title', '') == None:
			content = ''
		else:
			content = paper_dict.get('title', '') + " "
		content = paper_dict.get('abstractText', '')
		for section in paper_dict['sections']:
			if section['heading'] != None:
				content = content + " " +  section['text']
		content = re.sub("\n([0-9]*\n)+", "\n", content)
		return content





if __name__ == "__main__":
	path = sys.argv[1]
	ID = sys.argv[2]
	ckp_path = sys.argv[3]
	data = Prepare.fromJson(path, ID)
	ls = data.Embed()
	for rev in ls:
		(paper, review), rec, conf, rev_txt = rev
		model = REVSIGModel.Initialize(ckp_path)
		predictions = model.predict(paper, review)
		predictions['Actual REC'], predictions['Actual CONF'], predictions['comments'] = rec, conf, rev_txt
		print(predictions)
	
