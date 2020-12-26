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
from functools import partial
import matplotlib.pyplot as plt

logging.basicConfig(filename='add_extra_layer.log', filemode='w', level=logging.INFO)


class Vocab(object):
	def __init__(self):
		self.vocab = {'<sos>':1, '<pad>':0, '<eos>':2, '<unk>':3}
		self.count = {'<sos>':1, '<pad>':1, '<eos>':1, '<unk>':1}
		self.words = 4

	def Sentence(self, sentence, max_tokens=-1):
		numericalized = []
		for token in word_tokenize(sentence)[:max_tokens]:
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
		self.max_tokens = max_tokens

	def prepare(self, text):
		if not isinstance(self.vocab, dict):
			numericalized_text = self.vocab.Sentence(text, self.max_tokens)
		else:
			# print("In Else")
			numericalized_text = [self.vocab[token] if token in self.vocab.keys() else self.vocab['unk'] for token in word_tokenize(text)[:self.max_tokens]]

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
		self.rec_score = []

	def collect(self, example_score, type_=None):
		if type_ == 'sig':
			self.sig_scores.append(example_score)
		else:
			self.rec_score.append(example_score)

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
		return cls(p, r, Example.preprocess_scores(rec, stat), Example.preprocess_scores(conf), Example.preprocess_scores(sign, stat))

	@staticmethod
	def preprocess_text(text, lower=True):
		pass

	@staticmethod
	def preprocess_scores(score, stat=None):
		if isinstance(score, (str, int)):
			if stat != None:
				stat.collect(float(score))
			return float(score)
		elif isinstance(score, collections.Iterable):
			if stat != None:
				stat.collect(list(map(float, score)), type_='sig')
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
			paper_json_filenames_i = paper_json_filenames[i]
			random.shuffle(paper_json_filenames_i)
			for paper_json_filename in paper_json_filenames_i:
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
					if torch.sum(paper_content) != 0 and torch.sum(review_content) != 0:
						examples.append(Example.make(paper_content,review_content,review.RECOMMENDATION,review.CONFIDENCE,sig,stat).__dict__)
						# break
				# sys.exit()
		return cls(examples, vocab)

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)

def getLoaders(main_task_paths, scaffold_task_paths, batch_size=8, test=True):#, split=False):
	assert isinstance(main_task_paths, list)
	assert isinstance(scaffold_task_paths, list)

	# if normalizer != None:
	# stat_ = Stat()

	scaffold_task_dataset = sigData.readData(scaffold_task_paths)
	main_task_dataset = sigData.readData(main_task_paths, vocab = scaffold_task_dataset.vocab)
	vocab = scaffold_task_dataset.vocab
	if test:
		test_dataset = scaffold_task_dataset[:100]
		scaffold_task_dataset = scaffold_task_dataset[100:]

	# vocab = scaffold_task_dataset.vocab
	# if split:
	# 	test_dataset = scaffold_task_dataset[:100]
	# 	scaffold_task_dataset = scaffold_task_dataset[100:]

	


	# print(stat_.getStats())
	# print(stat_.rec_score)

	# plt.hist(stat_.rec_score, bins=10)
	# plt.savefig('stat.png')

	# print("Length of Sig_Scores: {}".format(len(normalizer_.sig_scores)))

	# assert len(scaffold_task_dataset.vocab) == len(main_task_dataset.vocab)

	main_task_len = len(main_task_dataset)
	scaffold_task_len = len(scaffold_task_dataset)
	# print(main_task_len, scaffold_task_len)

	#inflate the smaller dataset to match the size of the larger one
	if main_task_len < scaffold_task_len:
		difference = scaffold_task_len - main_task_len
		sample = [random.choice(main_task_dataset) for _ in range(difference)]
		main_task_dataset = main_task_dataset + sample

	main_task_dataloader = DataLoader(main_task_dataset, batch_size = batch_size, shuffle = True)
	scaffold_task_dataloader = DataLoader(scaffold_task_dataset, batch_size = batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
	



	#print(len(main_task_dataset), len(scaffold_task_dataset))
	#print(main_task_len, scaffold_task_len)
	# main_task_dataloader = None

	# if split:
	# 	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
	# 	return (main_task_dataloader, scaffold_task_dataloader, test_loader, vocab)
	return (main_task_dataloader, scaffold_task_dataloader, test_dataloader, vocab)



def num_params(list_):
	num = 0
	for i in list_:
		num += sum(p.numel() for p in i.parameters() if p.requires_grad)
	return num


def plot_stat(input=None, model=None):
	pass


class get_stat(object):
	def __init__(self, model):
		self.model = model
		self.act_means = [[] for _ in self.model.modules()]# if type(module) != nn.Sequential]
		self.act_stds = [[] for _ in self.model.modules()]# if type(module) != nn.Sequential]
		for i, m in enumerate(self.model.modules()):
			# if type(m) != nn.Sequential:
			m.register_forward_hook(partial(self.append_stats, i))

	def append_stats(self, i, mod, inp, outp):
		# print(inp)
		# inp=inp[0]
		self.act_means[i].append(outp.data.mean())
		self.act_stds[i].append(outp.data.std())

	def plot_stat(self, step):
		plt.figure()
		plt.subplot(121)
		plt.plot(range(len(self.act_means[2:])), [i[0].item() for i in self.act_means[2:]])
		plt.title('Means at ' + str(step))
		plt.subplot(122)
		plt.plot(range(len(self.act_means[2:])), [i[0].item() for i in self.act_stds[2:]])
		plt.title('Std at ' + str(step))
		plt.savefig('./Plots/' + str(step)+'.png')
		plt.close()

class decoder():
	def __init__(self, vocab):
		self.vocab = {v:k for k, v in vocab.items()}

	def decode(self, inp):
		# inp = inp.transpose(0,1)
		decoded_ = []
		for i in range(inp.shape[0]):
			decoded = [self.vocab[token] for token in inp[i].numpy().tolist() if self.vocab[token] != '<pad>']
			decoded_.append(" ".join(t for t in decoded))
		return decoded_




if __name__ == "__main__":

	main_task_loader, scaffold_task_loader, test_dataloader, vocab = getLoaders(['./Data/SignData/'], ['./Data/2018/'], batch_size=32)
	ntokens = len(vocab)
	print(ntokens)
	print("len main_task_loader: {} scaffold_task_loader: {} test_dataloader".format(len(main_task_loader),len(scaffold_task_loader), len(test_dataloader)))
	encoder = Encoder(ntokens, 256, 2, 256, 2)
	checkpoint = torch.load('checkpoint.pt')
	encoder.load_state_dict(checkpoint['encoder'])
	for param in encoder.parameters():
		param.requires_grad = False
	coder = reviewContext(32)
	# mainTask1 = MainTask1()
	# mainTask2 = MainTask2()
	# mainTask3 = MainTask3()
	recommendation_task = RecommendationTask()
	# print([module for module in recommendationTask.modules() if type(module) != nn.Sequential])
	# confidenceTask = ConfidenceTask()
	loss = nn.MSELoss()
	# model_params = list(encoder.parameters()) + list(coder.parameters())\
	# 							+ list(recommendationTask.parameters()) #+ list(confidenceTask.parameters())
								# + list(mainTask2.parameters()) + list(mainTask3.parameters()) + list(mainTask1.parameters())
	

	# for name, param in encoder.named_parameters():
	# 	print(name)

	for name, param in coder.named_parameters():
		print(name)

	for name, param in recommendation_task.named_parameters():
		print(name)

	# print(len(model_params))

	# print("Coder Parameters...")							
	# for param in coder.parameters():
	# 	print(param.shape)

	print("No. of model parameters: {}".format(num_params([coder, recommendation_task]))) #confidenceTask]))) #mainTask1,mainTask2,mainTask3,

	optimizer_recommendationTask = torch.optim.SGD(recommendation_task.parameters(), lr=0.01)
	optimizer_coder = torch.optim.SGD(coder.parameters(), lr=0.01)
	# optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.001)
	log_after = 20

	# fileobj = open("testing.txt", 'w')

	epoch_loss = 0.0

	best_val_loss = float('inf')

	decoder_ = decoder(vocab)
	step_ = 0

	for epoch in range(20):
		for step, data in enumerate(zip(main_task_loader, scaffold_task_loader), 0):
			main_data, scaffold_data = data
			bsz = scaffold_data['paper'].shape[0]

			main_paper, scaffold_paper = main_data['paper'], scaffold_data['paper']
			main_review, scaffold_review = main_data['review'], scaffold_data['review']

			with open("decoded.txt", "w") as f:
				d = decoder_.decode(scaffold_review)
				for i in d:
					f.write(i + '\n')

			# print(scaffold_paper.shape, scaffold_review.shape)

			#out_scaffold
			out_main, out_scaffold = encoder((main_paper.transpose(0,1), main_review.transpose(0,1)),\
											(scaffold_paper.transpose(0,1), scaffold_review.transpose(0,1)))

			# for i in out_scaffold:
			# 	print(i.shape)

			# main_rev_contexts = coder(out_main[1].transpose(0,1))
			# print(main_rev_contexts.shape)
			
			scaffold_rev_contexts = coder(out_scaffold[1].transpose(0,1))
			print(scaffold_rev_contexts.shape)
			
			# main_paper_repr = torch.mean(out_main[0].transpose(0,1), dim=1)
			# print(out_scaffold[0].shape)
			# scaffold_paper_repr = torch.mean(out_scaffold[0].transpose(0,1), dim=1)
			# print(scaffold_paper_repr.shape)
			
			# scaffold_review_repr = torch.mean(out_scaffold[1].transpose(0,1), dim=1)
			# print(scaffold_review_repr)

			# main_paper_repr = main_paper_repr.unsqueeze(1)
			# print(main_paper_repr.shape)

			# scaffold_paper_repr = scaffold_paper_repr.unsqueeze(-1)
			# print(scaffold_paper_repr.shape)

			# print((scaffold_rev_contexts*scaffold_paper_repr).shape)
			# print(torch.matmul(scaffold_rev_contexts, scaffold_paper_repr).shape)
			

			# main_temp = F.softmax(main_rev_contexts*main_paper_repr, dim=1)
			# print(main_temp.shape)
			# scaffold_temp = F.softmax(scaffold_rev_contexts*scaffold_paper_repr, dim=1)
			# print(scaffold_temp.shape)
			# scaffold_temp = torch.matmul(scaffold_rev_contexts, scaffold_paper_repr)
			# scaffold_temp = F.softmax(scaffold_temp, dim=1)



			# main_context = torch.sum(main_rev_contexts*main_temp, dim=1)
			# scaffold_context = torch.sum(scaffold_rev_contexts*scaffold_temp, dim=1)
			# print(scaffold_context.shape)
			# break
			# print(main_context.shape)
			# print(scaffold_context.shape)

			# rec_in = torch.cat([scaffold_context, scaffold_rev_contexts.view(8,-1)], dim = 1)
			# print(rec_in.shape)
			# break
			# scaffold_review_repr = torch.mean(out_scaffold[1].transpose(0,1), dim=1)
			# pred_rec = self.rec(scaffold_review_repr)
			rec_in = scaffold_rev_contexts.view(bsz,-1)
			print(rec_in.shape)

			stator = get_stat(recommendation_task)
			# pred_rec = recommendation_task(scaffold_review_repr)
			pred_rec = recommendation_task(rec_in)
			if step%log_after == 0:
				plot_stat(input=rec_in, model=recommendation_task)
				print(stator.act_means)
				stator.plot_stat(step)
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
			optimizer_coder.zero_grad()
			optimizer_recommendationTask.zero_grad()
			# optimizer_encoder.zero_grad()
			# loss_main1 = loss(pred_sign1.squeeze(1), main_data['sign'].float()[:,0])
			# loss_main2 = loss(pred_sign2.squeeze(1), main_data['sign'].float()[:,1])
			# loss_main3 = loss(pred_sign3.squeeze(1), main_data['sign'].float()[:,2])
			los = loss(pred_rec.squeeze(1), scaffold_data['recommendation'].float())
			# loss_conf = loss(pred_conf.squeeze(1), scaffold_data['confidence'].float())

			# logging.info('loss_main:{}, loss_rec:{}, loss_conf:{}'.format(loss_main.item(), loss_rec.item(), loss_conf.item()))

			# los = loss_main1 + loss_main2 + loss_main3 + loss_rec + loss_conf
			# los = loss_rec + loss_conf
			print(list(zip(pred_rec.squeeze(1),scaffold_data['recommendation'])))
			print(pred_rec.squeeze(1).shape, scaffold_data['recommendation'].float().shape)
			print(los.item())

			epoch_loss += los.item()

			los.backward()

			optimizer_recommendationTask.step()
			optimizer_coder.step()
			# optimizer_encoder.step()

			step_+=1

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
			# break
		with torch.no_grad():
			eval_loss = 0.0
			for j, data_t in enumerate(test_dataloader, 0):
				t_paper, t_review = data_t['paper'], data_t['review']
				bsz = t_paper.shape[0]

				_, out_t = encoder((None),\
											(t_paper.transpose(0,1), t_review.transpose(0,1)))

				rev_contexts_t = coder(out_t[1].transpose(0,1))
				print(rev_contexts_t.shape)

				rec_in_t = rev_contexts_t.view(bsz,-1)
				print(rec_in_t.shape)

				# stator = get_stat(recommendation_task)
				# pred_rec = recommendation_task(scaffold_review_repr)
				pred_rec_t = recommendation_task(rec_in_t)
				# if step%log_after == 0:
				# 	plot_stat(input=rec_in, model=recommendation_task)
				# 	print(stator.act_means)
				# 	stator.plot_stat(step)
				los_t = loss(pred_rec_t.squeeze(1), data_t['recommendation'].float())
				eval_loss += los_t.item()
				logging.info('EVALUATION')
				logging.info('Epoch: {}  Step:{} ###'.format(epoch, j))
				# logging.info('Loss: {}'.format(epoch_loss/log_after))
				# epoch_loss = 0.0
				#logging.info('loss_main1:{}, loss_main2:{}, loss_main3:{}, loss_rec:{}, loss_conf:{}'.format(loss_main1.item(),loss_main2.item(), loss_main3.item(), loss_rec.item(), loss_conf.item()))
				# logging.info('Rec: {}, Conf: {}, Sign1: {}, Sign2: {}, Sign3: {}'.format(list(zip(pred_rec,scaffold_data['recommendation'])), list(zip(pred_conf,scaffold_data['confidence'])),\
				# 			 list(zip(pred_sign1,main_data['sign'][:,0])),list(zip(pred_sign2,main_data['sign'][:,1])),\
				# 			list(zip(pred_sign3,main_data['sign'][:,2]))))

				logging.info('loss_rec:{}, loss_conf:{}'.format(los_t.item(), 0))#, loss_conf.item()))
				logging.info('Rec: {}, Conf: {}'.format(list(zip(pred_rec_t,data_t['recommendation'])),0))
			logging.info('Eval Epoch: {} Loss: {}'.format(epoch, eval_loss/len(test_dataloader)))

			if eval_loss/len(test_dataloader) < best_val_loss:
				best_val_loss = eval_loss/len(test_dataloader)
				torch.save({'encoder': encoder.state_dict(),
							'coder': coder.state_dict(),
							'recommendationTask': recommendation_task.state_dict()}, os.path.join('MODELS', 'testREC', 'ckpt' + str(epoch) + '.pt'))






