import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils_ import *
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

logging.basicConfig(filename='confidence.log', filemode = 'w', level=logging.INFO)



class BilinearAttention(nn.Module):
	def __init__(self):
		super(BilinearAttention, self).__init__()
		self.linear_proj = nn.Linear(128, 128, bias=False)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, paper, review):
		#paper, review = paper.transpose(0,2), review.transpose(0,2)
		print(paper.shape)
		print(review.shape)
		review = self.linear_proj(review.transpose(1,2)).transpose(1,2)
		print(review.shape)
		attn = self.softmax(torch.bmm(paper.transpose(1,2), review))
		print(attn.shape)
		print(attn.sum(dim=1, keepdim=True).shape)
		self.get_mask(attn)
	
	def get_mask(self, attn):
		a, b = torch.topk(attn, 5, dim = 1)
		print(a.shape, b.shape)
		print(b)

class MainTask(nn.Module):
	def __init__(self):
		super(MainTask, self).__init__()
		self.main = nn.Sequential(
						nn.Linear(256,128),
						nn.ReLU(),
						nn.Linear(128,64),
						nn.ReLU(),
						nn.Linear(64,32),
						nn.Linear(32,3)
						)
	
	def forward(self, output):
		return self.main(output)

class Recommendation_Task(nn.Module):
	def __init__(self):
		super(Recommendation_Task, self).__init__()
		self.recommendation = nn.Sequential(
						nn.Linear(256,128),
						nn.ReLU(),
						nn.Linear(128,64),
						nn.ReLU(),
						nn.Linear(64,1)
						# nn.Linear(32,1)
						)
	
	def forward(self, output):
		return self.recommendation(output)

class Confidence_Task(nn.Module):
	def __init__(self):
		super(Confidence_Task, self).__init__()
		self.confidence = nn.Sequential(
						nn.Linear(256,128),
						nn.ReLU(),
						nn.Linear(128,64),
						nn.ReLU(),
						nn.Linear(64,32),
						nn.Linear(32,1)
						)
	
	def forward(self, output):
		return self.confidence(output)
		


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		
		self.relu = nn.ReLU()
		self.prelu = nn.PReLU()
		self.convx1_first = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1)  #output size same as seq_len
		self.convx1_second = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
		self.convx5 = nn.Conv1d(in_channels=256, out_channels=128, padding=2, kernel_size=5) #output size same as seq_len
		#self.convx7 = nn.Conv1d(in_channels=512, out_channels=256, padding=3, kernel_size=7) #output size same as seq_len
		self.multihead_attn = nn.MultiheadAttention(128, 1)	
		#self.attention = BilinearAttention()
		
		#self.mainTask = MainTask()
		#self.recommendationTask = RecommendationTask()
		#self.confidenceTask = ConfidenceTask()

		#self.conv = nn.Conv1d(in_channels=256, out_channels=256,  kernel_size=779)
		#self.linear_ = nn.Linear(	
		#self.linear1 = nn.Linear(256,128)
		#self.linear2 = nn.Linear(128,64)
		#self.linear3 = nn.Linear(64,32)
		#self.linear4 = nn.Linear(32,1) 		
		
	def forward_once(self, paper, review):
		paper_downsampled = self.prelu(self.convx1_first(paper))
		review_downsampled = self.prelu(self.convx1_first(review))
		out_p = self.prelu(self.convx5(paper_downsampled))
		out_r = self.prelu(self.convx5(review_downsampled))
		#out = self.prelu(self.convx7(out_)) #size = (batch_size, emb_dim, seq_len) #attn shape = (seq_len, batch_size, emb_dim)
		attn, attn_ = self.multihead_attn(out_r.transpose(1,2).transpose(0,1), \
							out_p.transpose(1,2).transpose(0,1), \
							out_p.transpose(1,2).transpose(0,1)) 

		#self.attention(out_p, out_r)
		output = torch.cat([out_r, attn.transpose(0,1).transpose(1,2)], dim=1)
		output = self.convx1_second(output)
		output = torch.mean(output, dim=2)

		#main_preds = self.mainTask(output)
		#reco_preds = self.recommendationTask(output)
		#conf_preds = self.confidenceTask(output)
		#pred = self.relu(self.linear1(output))
		#pred = self.relu(self.linear2(pred))
		#pred = self.linear3(pred)
		#pred = self.linear4(pred)
		return output				#main_preds, reco_preds, conf_preds
	
	def forward(self, papers_sc, reviews_sc, papers_main, reviews_main):
		output_sc = self.forward_once(papers_sc, reviews_sc)
		output_main = self.forward_once(papers_main, reviews_main)
		return output_sc, output_main


if __name__ == "__main__":
	model = Model().to(device)
	mainTask = MainTask()
	recommendationTask = Recommendation_Task()
	confidenceTask = Confidence_Task()
	loss = nn.MSELoss()
	params = list(model.parameters()) + list(confidenceTask.parameters()) #list(recommendationTask.parameters())
	print(params)
	#params = [model.parameters(), mainTask.parameters(), recommendationTask.parameters(),confidenceTask.parameters()]
	optimizer = torch.optim.Adam(params)
	#ds = dataset.readData('./Data/2018', Transform())
	#print(len(ds))
	#loader = DataLoader(ds, batch_size=1, shuffle=True)
	loaders = getLoaders(batch_size=8)
	print("Length of Dataloaders: {} {}".format(len(loaders[0]), len(loaders[1])))
	epochs = 50
	pbar = range(epochs)
	for epoch in pbar:
		epoch_loss = []
		for i,d in enumerate(zip(*loaders),0):
			main_task_data = d[0]
			scaffold_task_data = d[1]
			papers_sc, reviews_sc, recs_sc, confs_sc = scaffold_task_data[0].transpose(1,2).float().to(device),\
							 scaffold_task_data[1].transpose(1,2).float().to(device), \
							 scaffold_task_data[2].float().to(device),\
							 scaffold_task_data[3].float().to(device)

			papers_main, reviews_main, sign_main = main_task_data[0].transpose(1,2).float().to(device),\
							 main_task_data[1].transpose(1,2).float().to(device), \
							 main_task_data[2].float().to(device)


			output_sc, output_main = model(papers_sc, reviews_sc, papers_main, reviews_main)
			#print(output_sc.shape)
			#print(output_main.shape)
						
			main_preds = None #mainTask(output_main)
			#print(main_preds.shape)
			rec_preds = None #recommendationTask(output_sc)
			#print(rec_preds.shape)
			conf_preds = confidenceTask(output_sc)
			#print(conf_preds.shape)
			#print(sign_main.shape)
			loss_main = None #loss(main_preds, sign_main)
			#print(loss_main)
			loss_rec = None #loss(rec_preds.squeeze(1), recs_sc)
			#print(loss_rec)
			loss_conf = loss(conf_preds.squeeze(1), confs_sc)
			#print(loss_conf)	
			optimizer.zero_grad()
			los = loss_conf #loss_rec #loss_main + loss_rec + loss_conf
			los.backward()
			epoch_loss.append(los.to('cpu').item())
			optimizer.step()
			# print(list(zip(rec_preds, recs_sc)))
		print('Epoch: {} Loss: {}'.format(epoch, np.average(epoch_loss)))
		logging.info('Main loss: {}, Rec loss: {}, Conf loss: {}'.format(str(loss_main), str(loss_rec), str(loss_conf)))
		logging.info('MainTask: {} {}'.format(str(main_preds),str(sign_main)))
		logging.info('Recommendation: {} {}'.format(str(rec_preds),str(recs_sc)))
		logging.info('Confidence: {} {}'.format(str(conf_preds),str(confs_sc)))
		logging.info("Epoch: {} Training Loss: {}".format(str(epoch), str(np.average(epoch_loss))))
	
