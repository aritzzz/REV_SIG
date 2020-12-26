import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Transform(object):
	def __init__(self):
		pass
	def __call__(self, array, max_sents):
		if max_sents < array.shape[0]:
			return torch.from_numpy(array[:max_sents,:])
		else:
			return torch.from_numpy(np.pad(array, [(0, max_sents - array.shape[0]), (0,0)], mode = 'constant', constant_values = 0.0))

class ScaleSigScores(object):
	def __init__(self):
		pass
	def __call__(self, array, min_, max_):
		sent = array[2]
		array = (array-min_)/(max_ - min_)
		array = (array*9) + 1
		array[2] = (((sent - (-1))*9)/2) + 1
		return array	

class jsonEncoder(object):
	def __init__(self, json_obj=None, mode = None):
		self.json_obj = json_obj
		self.mode = mode

	@classmethod
	def from_json(cls, path, review_filename, mode):
		try:
			return cls(open(os.path.join(path, 'Embeddings', review_filename)), mode=mode)
		except FileNotFoundError:
			return cls(None)

	def __call__(self):
		if not self.json_obj == None:
			encoded = json.load(self.json_obj)
			paper = np.asarray(encoded['paper'])
			reviews = []
			if self.mode == 'SCAFFOLDS':
				rec_score = []
				conf_score = []
				for i, review in enumerate(encoded['reviews']):
					reviews.append(np.asarray(review.get('review_text')))
					rec_score.append(review.get('RECOMMENDATION'))
					conf_score.append(review.get('CONFIDENCE'))		
				return paper, reviews, rec_score, conf_score
			elif self.mode == 'MAIN':
				significance_score = []
				for i, review in enumerate(encoded['reviews']):
					reviews.append(np.asarray(review.get('review_text')))
					significance_score.append([float(i) for i in review.get('SIGNIFICANCE')])
				return paper, reviews, significance_score
			else:
				rec_score = []
				conf_score = []
				significance_score = []
				for i, review in enumerate(encoded['reviews']):
					reviews.append(np.asarray(review.get('review_text')))
					rec_score.append(review.get('RECOMMENDATION'))
					conf_score.append(review.get('CONFIDENCE'))	
					significance_score.append([float(i) for i in review.get('SIGNIFICANCE')])
				return paper, reviews, rec_score, conf_score, significance_score
		else:
			return None
		

class dataset(Dataset):
	def __init__(self, data, transform = None, max_paper_sentences = None, max_review_sentences = None, mode = 'SCAFFOLDS'):
		self.data = data
		self.max_paper_sentences = max_paper_sentences
		self.max_review_sentences = max_review_sentences
		self.transform = transform
		self.mode = mode


	@classmethod
	def readData(cls, path, transform = None, jsonEncoder = jsonEncoder(), mode='SCAFFOLDS', n=100):
		reviews_dir = os.listdir(os.path.join(path, 'reviews'))[:n]
		papers, reviews, rec_scores, conf_scores, sig_scores = [], [], [], [], []
		max_paper_sents = 0
		max_review_sents = 0
		pbar = tqdm(reviews_dir)
		for review_dir in pbar:
			#print(review_dir)
			pbar.set_description("Reading Embeddings...")
			ret = jsonEncoder.from_json(path, review_dir, mode=mode)()
			if ret == None:
				continue
			if ret[0].shape[0] > max_paper_sents:
				max_paper_sents = ret[0].shape[0]
			for i, rev in enumerate(ret[1],0):
				papers.append(ret[0])
				reviews.append(rev)
				if mode == 'SCAFFOLDS':
					rec_scores.append(int(ret[2][i]))
					conf_scores.append(int(ret[3][i]))
				elif mode == 'MAIN':
					sig_scores.append((ret[2][i]))
				else:
					rec_scores.append(int(ret[2][i]))
					conf_scores.append(int(ret[3][i]))
					sig_scores.append((ret[4][i]))

				if rev.shape[0] > max_review_sents:
					max_review_sents = rev.shape[0]	

		if mode == 'SCAFFOLDS':		
			return cls((papers,reviews, rec_scores, conf_scores), transform, max_paper_sents, max_review_sents, mode=mode)
		if mode == 'MAIN':
			return cls((papers,reviews, np.asarray(sig_scores)), transform, max_paper_sents, max_review_sents, mode = mode)
		if mode == 'TEST':
			return cls((papers,reviews, rec_scores, conf_scores, np.asarray(sig_scores)), transform, max_paper_sents, max_review_sents, mode = mode)
		

	def __getitem__(self, index):
		if self.mode == 'SCAFFOLDS':
			if self.transform:
				return self.transform(self.data[0][index], self.max_paper_sentences), \
					self.transform(self.data[1][index], self.max_review_sentences), \
					self.data[2][index], self.data[3][index]
			else:
				return self.data[0][index],self.data[1][index],self.data[2][index],self.data[3][index]
		elif self.mode == 'MAIN':
			if self.transform:
				return self.transform(self.data[0][index], self.max_paper_sentences), \
					self.transform(self.data[1][index], self.max_review_sentences), \
					self.data[2][index]
			else:
				return self.data[0][index],self.data[1][index],self.data[2][index]
		else:
			if self.transform:
				return self.transform(self.data[0][index], self.max_paper_sentences), \
					self.transform(self.data[1][index], self.max_review_sentences), \
					self.data[2][index], self.data[3][index], self.data[4][index]
			else:
				return self.data[0][index],self.data[1][index],self.data[2][index], self.data[3][index], self.data[4][index]
	def __len__(self):
		return len(self.data[0])


class RevSigData(Dataset):
	def __init__(self, path, mode = 'SCAFFOLDS', slice_=-1, transform = None, sigtx = None):
		self.path = path
		self.slice=slice_
		self.data = os.listdir(path)[:self.slice]
		self.mode = mode
		if self.mode == 'SCAFFOLDS':
			self.max_paper_sentences, self.max_review_sentences = 790, 790 #self.forTransform()
		else:
			self.max_paper_sentences, self.max_review_sentences, self.sig_min, self.sig_max = 790, 790, np.array([0.05428571,0.93013972,-0.99966815]), np.array([33.26288336,252.25067107,0.99966679])#self.forTransform()
		
		self.transform = transform
		self.SigTransform = sigtx
		
	def forTransform(self):
		max_paper_sents = 0
		max_review_sents = 0
		if self.mode != 'SCAFFOLDS':
			sig_scores = []
			
		for file in self.data:
			json_obj = open(os.path.join(self.path, file))
			ret = json.load(json_obj)
			if len(ret['paper']) > max_paper_sents:
				max_paper_sents = len(ret['paper'])
			if len(ret['review']) > max_review_sents:
				max_review_sents = len(ret['review'])
			if self.mode != 'SCAFFOLDS':
				sig_scores.append(ret['significance'])
		if self.mode != 'SCAFFOLDS':            
			return max_paper_sents, max_review_sents,np.asarray(sig_scores).min(axis=0),np.asarray(sig_scores).max(axis=0)
		else:
			return max_paper_sents, max_review_sents
		
	def __getitem__(self, index):
		file = self.data[index]
		json_obj = open(os.path.join(self.path, file))
		json_data = json.load(json_obj)

		if self.mode == 'SCAFFOLDS':
			if self.transform:
				return self.transform(np.asarray(json_data['paper']), self.max_paper_sentences), \
					self.transform(np.asarray(json_data['review']), self.max_review_sentences), \
					json_data['recommendation'], json_data['confidence']
			else:
				return json_data['paper'], json_data['review'], json_data['recommendation'], json_data['confidence']
		elif self.mode == 'MAIN':
			if self.transform:
				return self.transform(np.asarray(json_data['paper']), self.max_paper_sentences), \
					self.transform(np.asarray(json_data['review']), self.max_review_sentences), \
					self.SigTransform(np.asarray(json_data['significance']), self.sig_min, self.sig_max)
			else:
				return json_data['paper'],json_data['review'],json_data['significance']
		else:
			if self.transform:
				return self.transform(np.asarray(json_data['paper']), self.max_paper_sentences), \
					self.transform(np.asarray(json_data['review']), self.max_review_sentences), \
					json_data['recommendation'], json_data['confidence'], self.SigTransform(np.asarray(json_data['significance']), self.sig_min, self.sig_max)
			else:
				return json_data['paper'], json_data['review'],json_data['recommendation'], json_data['confidence'],json_data['significance'] 
	def __len__(self):
		return len(self.data)


# class RevSigData(Dataset):
#     def __init__(self, path, mode = 'SCAFFOLDS', slice_=-1, transform = None):
#         self.path = path
#         self.slice=slice_
#         self.data = os.listdir(path)[:self.slice]
#         self.mode = mode
#         self.max_paper_sentences, self.max_review_sentences = 790,790#self.forTransform()
#         self.transform = transform

		
#     def forTransform(self):
#         max_paper_sents = 0
#         max_review_sents = 0
#         for file in self.data:
#             json_obj = open(os.path.join(self.path, file))
#             ret = json.load(json_obj)
#             if len(ret['paper']) > max_paper_sents:
#                 max_paper_sents = len(ret['paper'])
#             if len(ret['review']) > max_review_sents:
#                 max_review_sents = len(ret['review'])
#         return max_paper_sents, max_review_sents

#     def __getitem__(self, index):
#         file = self.data[index]
#         json_obj = open(os.path.join(self.path, file))
#         json_data = json.load(json_obj)

#         if self.mode == 'SCAFFOLDS':
#             if self.transform:
#                 return self.transform(np.asarray(json_data['paper']), self.max_paper_sentences), \
#                     self.transform(np.asarray(json_data['review']), self.max_review_sentences), \
#                     json_data['recommendation'], json_data['confidence']
#             else:
#                 return json_data['paper'], json_data['review'], json_data['recommendation'], json_data['confidence']
#         elif self.mode == 'MAIN':
#             if self.transform:
#                 return self.transform(np.asarray(json_data['paper']), self.max_paper_sentences), \
#                     self.transform(np.asarray(json_data['review']), self.max_review_sentences), \
#                     np.asarray(json_data['significance'])
#             else:
#                 return json_data['paper'],json_data['review'],json_data['significance']
#         else:
#             if self.transform:
#                 return self.transform(np.asarray(json_data['paper']), self.max_paper_sentences), \
#                     self.transform(np.asarray(json_data['review']), self.max_review_sentences), \
#                     json_data['recommendation'], json_data['confidence'], np.asarray(json_data['significance'])
#             else:
#                 return json_data['paper'], json_data['review'],json_data['recommendation'], json_data['confidence'],json_data['significance'] 
#     def __len__(self):
#         return len(self.data)




def getLoaders(main_task_path = './Data/SignData/train_data', scaffold_task_path = './Data/2018/train_data', batch_size=8, slice=[-1, -1, -1], test_path='./Data/SignData/test_data'):
	print('Reading the Main Task Dataset...')
	main_task_dataset = RevSigData(main_task_path, mode='MAIN', slice_=slice[0], transform=Transform(), sigtx=ScaleSigScores())
	#main_task_dataset = dataset.readData(main_task_path, Transform(), mode='MAIN', n=slice[0])
	print('Reading the Scaffolds Task Dataset...')
	scaffold_task_dataset = RevSigData(scaffold_task_path, mode='SCAFFOLDS', slice_=slice[1], transform=Transform())
	#scaffold_task_dataset = dataset.readData(scaffold_task_path, Transform(), mode='SCAFFOLDS', n=slice[1])
	

	if test_path:
		print('Reading the test Dataset')
		test_dataset = RevSigData(test_path, mode='TEST', slice_=slice[2], transform=Transform(), sigtx=ScaleSigScores())
		#test_dataset = dataset.readData(test_path, Transform(), mode='TEST', n=slice[2])
	else:
		test_dataset = None


	#length of the both task datasets
	main_task_len = len(main_task_dataset)
	scaffold_task_len = len(scaffold_task_dataset)
	test_len = len(test_dataset)

	#inflate the smaller dataset to match the size of the larger one
	if main_task_len < scaffold_task_len:
		difference = scaffold_task_len - main_task_len
		sample = [random.choice(main_task_dataset) for _ in range(difference)]
		main_task_dataset = main_task_dataset + sample
	
	main_task_dataloader = DataLoader(main_task_dataset, batch_size = batch_size, shuffle = True, num_workers=4)
	scaffold_task_dataloader = DataLoader(scaffold_task_dataset, batch_size = batch_size, shuffle=True, num_workers=4)
	if test_dataset != None:
		test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	else:
		test_data_loader = None
	
	# print(len(main_task_dataset), len(scaffold_task_dataset))
	#print(main_task_len, scaffold_task_len)
	return (main_task_dataloader, scaffold_task_dataloader, test_data_loader)
	

	


if __name__ == "__main__":
	mainloader, scaffoldloader, testloader = getLoaders(batch_size=2, slice=[5,5,5])
	print(len(mainloader), len(scaffoldloader), len(testloader))
	for i, d in enumerate(zip(*(mainloader, scaffoldloader, testloader)), 0):
		main, scaffold, test = d
		print(main)
		print(scaffold)
		print(test)
		break
		
