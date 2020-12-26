from Review import Review
from Paper import Paper
from ScienceParse import ScienceParse
from ScienceParseReader import ScienceParseReader
import sys
import os
import glob
import json
import random
import torch
import numpy as np
from nltk import word_tokenize
import collections
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence

from transformer import *
import torch.nn.functional as F
import logging

logging.basicConfig(filename='add_extra_layer.log', filemode='w', level=logging.INFO)


class Vocab(object):
	def __init__(self):
		self.vocab = {'<sos>':1, '<pad>':0, '<eos>':2, '<unk>':3}
		self.count = {'<sos>':1, '<pad>':1, '<eos>':1, '<unk>':1}
		self.words = 4

	def Sentence(self, sentence, max_tokens=-1):
		numericalized = []
		for token in word_tokenize(sentence.lower())[:max_tokens]:
			numericalized.append(self.Word(token))
		return numericalized

	def Word(self, token):
		if token not in self.vocab.keys():
			self.vocab[token] = self.words
			self.words+=1
			self.count[token] = 1
			return self.vocab[token]
		else:
			self.count[token]+=1
			return self.vocab[token]

	def filter(self, threshold=0):
		return {k: v for k, v in self.vocab.items() if self.count[k] > threshold or k in ['<sos>', '<pad>', '<eos>', '<unk>']}



class Preprocessor(object):
	def __init__(self, vocab, truncate_text=True, max_tokens=512):
		self.vocab = vocab
		self.truncate_text = truncate_text
		self.max_tokens = max_tokens + 2

	def prepare(self, text):
		if not isinstance(self.vocab, dict):
			numericalized_text = [1] + self.vocab.Sentence(text, self.max_tokens-2) + [2]
		else:
			# print("In Else")
			numericalized_text = [self.vocab[token] if token in self.vocab.keys() else self.vocab['unk'] for token in word_tokenize(text.lower())[:self.max_tokens]]

		if len(numericalized_text) < self.max_tokens:
			numericalized_text.extend([0]*(self.max_tokens - len(numericalized_text)))
			assert len(numericalized_text) == self.max_tokens
			return torch.LongTensor(numericalized_text)

		if self.truncate_text:
			# print("In truncate!")
			numericalized_text = numericalized_text[:self.max_tokens]
			assert len(numericalized_text) == self.max_tokens
			return torch.LongTensor(numericalized_text)

class Stat(object):

	def __init__(self):
		self.sig_scores = []

	def collect(self, example_score):
		self.sig_scores.append(example_score)

	def getStats(self):
		self.sig_scores = np.array(self.sig_scores)
		print(self.sig_scores.shape)
		return np.amax(self.sig_scores, axis=0), np.amin(self.sig_scores, axis=0)






class Example(object):
	def __init__(self, paper, review, recommendation, confidence, sign):
		self.paper = paper
		self.review = review
		self.recommendation = recommendation
		self.confidence = confidence
		self.sign = sign

	@classmethod
	def make(cls, p, r, rec, conf, sign, stat):
		return cls(p, r, Example.preprocess_scores(rec), Example.preprocess_scores(conf), Example.preprocess_scores(sign, stat))

	@staticmethod
	def preprocess_text(text, lower=True):
		pass

	@staticmethod
	def preprocess_scores(score, stat=None):
		if isinstance(score, (str, int)):
			return float(score)
		elif isinstance(score, collections.Iterable):
			if stat != None:
				stat.collect(list(map(float, score)))
			return torch.tensor(list(map(float, score)))


class sigData(Dataset):
	def __init__(self, data, vocab):
		self.data = data
		self.vocab = vocab if isinstance(vocab, dict) else vocab.vocab

		assert isinstance(self.vocab, dict)

	@classmethod
	def readData(cls, paths, vocab=None, truncate=True, stat=None):
		""" paths is a list of path """
		review_dirs = [os.path.join(path,'reviews/') for path in paths]
		print(review_dirs)
		scienceparse_dirs = [os.path.join(path, 'parsed_pdfs/') for path in paths]
		print(scienceparse_dirs)
		paper_json_filenames = [glob.glob('{}/*.json'.format(review_dir)) for review_dir in review_dirs]

		assert len(review_dirs) == len(scienceparse_dirs) == len(paper_json_filenames) == len(paths)

		if vocab == None:
			print("Initialize Vocab object...")
			vocab = Vocab()
			print(type(vocab))
		else:
			print("Use the saved vocab...")
			assert isinstance(vocab, dict)

		examples = []

		preprocessor = Preprocessor(vocab)

		for i, path in enumerate(paths):
			for paper_json_filename in paper_json_filenames[i]:
				paper = Paper.from_json(paper_json_filename)
				paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID,paper.TITLE,paper.ABSTRACT,scienceparse_dirs[i])
				if paper.SCIENCEPARSE == None:
					continue
				paper_content = preprocessor.prepare(paper.SCIENCEPARSE.get_paper_content())
				# print(paper_content.shape)
				# print(torch.sum(paper_content[1:-1]))
				for review in paper.REVIEWS:
					review_content = preprocessor.prepare(review.COMMENTS)
					sig = review.SIGNIFICANCE_SCORES.split(":")
					if torch.sum(paper_content[1:-1]) != 0 and torch.sum(review_content[1:-1]) != 0:
						examples.append(Example.make(paper_content,review_content,review.RECOMMENDATION,review.CONFIDENCE,sig,stat).__dict__)
						# break
				# sys.exit()
		return cls(examples, vocab)

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)

def getLoaders(main_task_paths, scaffold_task_paths, batch_size=8, split=False):
	assert isinstance(main_task_paths, list)
	assert isinstance(scaffold_task_paths, list)

	# if normalizer != None:
	# stat_ = Stat()

	scaffold_task_dataset = sigData.readData(scaffold_task_paths)
	vocab = scaffold_task_dataset.vocab
	if split:
		test_dataset = scaffold_task_dataset[:100]
		scaffold_task_dataset = scaffold_task_dataset[100:]

	# main_task_dataset = sigData.readData(main_task_paths, vocab = scaffold_task_dataset.vocab, stat=stat_)

	# print(stat_.getStats())

	# print("Length of Sig_Scores: {}".format(len(normalizer_.sig_scores)))

	# assert len(scaffold_task_dataset.vocab) == len(main_task_dataset.vocab)

	# main_task_len = len(main_task_dataset)
	# scaffold_task_len = len(scaffold_task_dataset)

	#inflate the smaller dataset to match the size of the larger one
	# if main_task_len < scaffold_task_len:
	# 	difference = scaffold_task_len - main_task_len
	# 	sample = [random.choice(main_task_dataset) for _ in range(difference)]
	# 	main_task_dataset = main_task_dataset + sample

	# main_task_dataloader = DataLoader(main_task_dataset, batch_size = batch_size, shuffle = True)
	scaffold_task_dataloader = DataLoader(scaffold_task_dataset, batch_size = batch_size, shuffle=True)
	



	#print(len(main_task_dataset), len(scaffold_task_dataset))
	#print(main_task_len, scaffold_task_len)
	main_task_dataloader = None

	if split:
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
		return (main_task_dataloader, scaffold_task_dataloader, test_loader, vocab)
	return (main_task_dataloader, scaffold_task_dataloader, len(scaffold_task_dataset.vocab))



def num_params(list_):
	num = 0
	for i in list_:
		num += sum(p.numel() for p in i.parameters() if p.requires_grad)
	return num




if __name__ == "__main__":

	main_task_loader, scaffold_task_loader, ntokens = getLoaders(['./Data/SignData/'], ['./Data/2018/'], batch_size=8)
	print("len main_task_loader: {} scaffold_task_loader: {}".format(len(main_task_loader),len(scaffold_task_loader)))
	encoder = Encoder(ntokens, 256, 2, 256, 2)
	coder = reviewContext(3)
	# mainTask1 = MainTask1()
	# mainTask2 = MainTask2()
	# mainTask3 = MainTask3()
	recommendationTask = RecommendationTask()
	# confidenceTask = ConfidenceTask()
	loss = nn.MSELoss()
	model_params = list(encoder.parameters()) + list(coder.parameters())\
								+ list(recommendationTask.parameters()) #+ list(confidenceTask.parameters())
								# + list(mainTask2.parameters()) + list(mainTask3.parameters()) + list(mainTask1.parameters())
	
	print("Coder Parameters...")							
	for param in coder.parameters():
		print(param.shape)

	print("No. of model parameters: {}".format(num_params([encoder,coder,recommendationTask]))) #confidenceTask]))) #mainTask1,mainTask2,mainTask3,

	optimizer = torch.optim.Adam(model_params, 0.01)
	log_after = 20

	# fileobj = open("testing.txt", 'w')

	epoch_loss = 0.0

	for epoch in range(20):
		
		for step, data in enumerate(zip(main_task_loader, scaffold_task_loader), 0):
			main_data, scaffold_data = data

			main_paper, scaffold_paper = main_data['paper'], scaffold_data['paper']
			main_review, scaffold_review = main_data['review'], scaffold_data['review']

			# print(scaffold_paper.shape, scaffold_review.shape)


			out_main, out_scaffold = encoder((main_paper.transpose(0,1), main_review.transpose(0,1)),\
											(scaffold_paper.transpose(0,1), scaffold_review.transpose(0,1)))

			# for i in out_scaffold:
			# 	print(i.shape)

			# main_rev_contexts = coder(out_main[1].transpose(0,1))
			# print(main_rev_contexts.shape)
			
			scaffold_rev_contexts = coder(out_scaffold[1].transpose(0,1))
			# print(scaffold_rev_contexts.shape)
			
			# main_paper_repr = torch.mean(out_main[0].transpose(0,1), dim=1)
			# print(out_scaffold[0].shape)
			scaffold_paper_repr = torch.mean(out_scaffold[0].transpose(0,1), dim=1)
			# print(scaffold_paper_repr.shape)
			


			# main_paper_repr = main_paper_repr.unsqueeze(1)
			# print(main_paper_repr.shape)

			scaffold_paper_repr = scaffold_paper_repr.unsqueeze(-1)
			# print(scaffold_paper_repr.shape)

			# print((scaffold_rev_contexts*scaffold_paper_repr).shape)
			# print(torch.matmul(scaffold_rev_contexts, scaffold_paper_repr).shape)
			

			# main_temp = F.softmax(main_rev_contexts*main_paper_repr, dim=1)
			# print(main_temp.shape)
			# scaffold_temp = F.softmax(scaffold_rev_contexts*scaffold_paper_repr, dim=1)
			# print(scaffold_temp.shape)
			scaffold_temp = torch.matmul(scaffold_rev_contexts, scaffold_paper_repr)
			scaffold_temp = F.softmax(scaffold_temp, dim=1)



			# main_context = torch.sum(main_rev_contexts*main_temp, dim=1)
			scaffold_context = torch.sum(scaffold_rev_contexts*scaffold_temp, dim=1)
			# print(scaffold_context.shape)
			# break
			# print(main_context.shape)
			# print(scaffold_context.shape)

			rec_in = torch.cat([scaffold_context, scaffold_rev_contexts.view(8,-1)], dim = 1)
			# print(rec_in.shape)
			# break

			pred_rec = recommendationTask(rec_in)
			# pred_conf = confidenceTask(scaffold_context)
			# pred_sign1 = mainTask1(main_context)
			# pred_sign2 = mainTask2(main_context)
			# pred_sign3 = mainTask3(main_context)

			# for i in (pred_rec, pred_conf, pred_sign):
			# 	print(dtype(i))


			# for i in (main_data['sign'], scaffold_data['recommendation'], scaffold_data['confidence']):
			# 	print(i)
			# 	print(i.shape, type(i))
			# break
			optimizer.zero_grad()
			# loss_main1 = loss(pred_sign1.squeeze(1), main_data['sign'].float()[:,0])
			# loss_main2 = loss(pred_sign2.squeeze(1), main_data['sign'].float()[:,1])
			# loss_main3 = loss(pred_sign3.squeeze(1), main_data['sign'].float()[:,2])
			los = loss(pred_rec.squeeze(1), scaffold_data['recommendation'].float())
			# loss_conf = loss(pred_conf.squeeze(1), scaffold_data['confidence'].float())

			# logging.info('loss_main:{}, loss_rec:{}, loss_conf:{}'.format(loss_main.item(), loss_rec.item(), loss_conf.item()))

			# los = loss_main1 + loss_main2 + loss_main3 + loss_rec + loss_conf
			# los = loss_rec + loss_conf

			epoch_loss += los.item()

			los.backward()

			optimizer.step()

			# logging.info(los.item())
			if torch.isnan(los):
				print(main_data)
				print(scaffold_data)

			if (step+1)%log_after == 0:
				logging.info('Epoch: {}  Step:{} ###'.format(epoch, step))
				logging.info('Loss: {}'.format(epoch_loss/log_after))
				epoch_loss = 0.0
				#logging.info('loss_main1:{}, loss_main2:{}, loss_main3:{}, loss_rec:{}, loss_conf:{}'.format(loss_main1.item(),loss_main2.item(), loss_main3.item(), loss_rec.item(), loss_conf.item()))
				# logging.info('Rec: {}, Conf: {}, Sign1: {}, Sign2: {}, Sign3: {}'.format(list(zip(pred_rec,scaffold_data['recommendation'])), list(zip(pred_conf,scaffold_data['confidence'])),\
				# 			 list(zip(pred_sign1,main_data['sign'][:,0])),list(zip(pred_sign2,main_data['sign'][:,1])),\
				# 			list(zip(pred_sign3,main_data['sign'][:,2]))))

				logging.info('loss_rec:{}, loss_conf:{}'.format(los.item(), 0))#, loss_conf.item()))
				logging.info('Rec: {}, Conf: {}'.format(list(zip(pred_rec,scaffold_data['recommendation'])),0))#, list(zip(pred_conf,scaffold_data['confidence']))))
