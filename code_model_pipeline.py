from data_utils_ import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from collections import OrderedDict
import argparse
import numpy as np
from Models import Pipeline, MTLoss, Prediction #CrossAttention, Context
device ="cuda"
torch.cuda.empty_cache()
import gc

#logging.basicConfig(filename='code_model_pipeline_3codes.log', filemode='w', level=logging.INFO)


args = None



def evaluate(model, main_task_predictor, scaffold_task_predictor, Criterion, test_loader):
	with torch.no_grad():
		eval_loss = []
		mseLoss_Ex = []
		mseLoss_Subj = []
		mseLoss_Int = []
		mseLoss_Rec = []
		mseLoss_Conf = []
		mseloss = nn.MSELoss()
		for i, d in enumerate(test_loader,0):
			scaffold_task_data = d
			papers_sc, reviews_sc, recs_sc, confs_sc, sign_m = scaffold_task_data[0].transpose(1,2).float().to(device),\
							 scaffold_task_data[1].transpose(1,2).float().to(device), \
							 scaffold_task_data[2].float().to(device),\
							 scaffold_task_data[3].float().to(device),\
							 scaffold_task_data[4].float().to(device)

			ex, subj, opine = sign_m[:,0], sign_m[:,1], sign_m[:,2]
			out, rec_codes, conf_codes = model(papers_sc, reviews_sc)
			rec_preds, conf_preds = scaffold_task_predictor(rec_codes.view(out.shape[0], -1), conf_codes.view(out.shape[0], -1))

			# out_m, rec_codes_m, conf_codes_m = model(papers_sc, reviews_sc)
			ex_preds, subj_preds, intensity_preds = main_task_predictor(out, rec_codes, conf_codes)


			loss = Criterion([rec_preds.squeeze(1), conf_preds.squeeze(1), ex_preds.squeeze(1), subj_preds.squeeze(1), intensity_preds.squeeze(1)], [recs_sc, confs_sc, ex, subj, opine])
			
			eval_loss.append(loss.item())

			ex = mseloss(ex_preds.squeeze(1), ex)
			mseLoss_Ex.append(ex.item())
			subj = mseloss(subj_preds.squeeze(1), subj)
			mseLoss_Subj.append(subj.item())
			inte = mseloss(ex_preds.squeeze(1), opine)
			mseLoss_Int.append(inte.item())
			recl = mseloss(rec_preds.squeeze(1), recs_sc)
			mseLoss_Rec.append(recl.item())
			confl = mseloss(conf_preds.squeeze(1), confs_sc)
			mseLoss_Conf.append(confl.item())

		return np.average(eval_loss), np.average(mseLoss_Ex), np.average(mseLoss_Subj), np.average(mseLoss_Int), np.average(mseLoss_Rec), np.average(mseLoss_Conf)



def train():
	main_task_loader, scaffold_task_loader, test_loader = getLoaders(batch_size=args.batch_size, slice=[-1,-1,-1])
	model = Pipeline.Pipeline(args).to(device)
	main_task_predictor = Prediction.MainPrediction(args.upscale_dim, args.upscale_dim, 32).to(device)
	scaffold_task_predictor = Prediction.ScaffoldPrediction(args.upscale_dim, 8).to(device)

	print(model)
	for name, param in model.named_parameters():
		print(name, param.shape)
	print("No. of Trainable parameters {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

	Criterion = MTLoss.MTLoss().to(device)
	optimizer = torch.optim.Adam(list(model.parameters()) + list(Criterion.parameters()) + list(main_task_predictor.parameters()) + list(scaffold_task_predictor.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
	epochs = 100
	best_eval_loss = np.inf

	if not os.path.exists(args.model_dir):
		os.mkdir(args.model_dir)
	MODEL_DIR = args.model_dir

	for epoch in range(epochs):
		model.train()
		epoch_loss = []
		for i, d in enumerate(zip(scaffold_task_loader, main_task_loader),0):
			#print(i)
			main_task_data = d[1]
			scaffold_task_data = d[0]
			papers_sc, reviews_sc, recs_sc, confs_sc = scaffold_task_data[0].transpose(1,2).float().to(device),\
								 scaffold_task_data[1].transpose(1,2).float().to(device), \
								 scaffold_task_data[2].float().to(device),\
								 scaffold_task_data[3].float().to(device)

			papers_m, reviews_m, sign_m = main_task_data[0].transpose(1,2).float().to(device),\
								 main_task_data[1].transpose(1,2).float().to(device), \
								 main_task_data[2].float().to(device)

			ex, subj, opine = sign_m[:,0], sign_m[:,1], sign_m[:,2]

			#print(ex.shape, subj.shape, opine.shape, recs_sc.shape, confs_sc.shape)

			optimizer.zero_grad()
			out, rec_codes, conf_codes = model(papers_sc, reviews_sc)
			rec_preds, conf_preds = scaffold_task_predictor(rec_codes.view(out.shape[0], -1), conf_codes.view(out.shape[0], -1))


			#do the for the main task
			out_m, rec_codes_m, conf_codes_m = model(papers_m, reviews_m)
			ex_preds, subj_preds, intensity_preds = main_task_predictor(out_m, rec_codes_m, conf_codes_m)
			#print(ex_preds.shape, subj_preds.shape, intensity_preds.shape)


			loss = Criterion([rec_preds.squeeze(1), conf_preds.squeeze(1), ex_preds.squeeze(1), subj_preds.squeeze(1), intensity_preds.squeeze(1)], [recs_sc, confs_sc, ex, subj, opine])
			epoch_loss.append(loss.item())
			loss.backward()
			optimizer.step()
		#print("Epoch {} Loss: {:.3f}".format(epoch, np.average(epoch_loss)))
			del papers_sc
			del reviews_sc
			gc.collect()
		# 	break
		# break

		with torch.no_grad():
			eval_loss, ex, subj, inte, rec, conf = evaluate(model, main_task_predictor, scaffold_task_predictor, Criterion, test_loader)
		
			print('Epoch: {} Train Loss: {:.6f}, Test Loss: {:.6f}'.format(epoch, np.average(epoch_loss),\
							eval_loss))
			print('Exhaustive: {:.6f} Subjectivity: {:.6f}, Intensity {:.6f}, Recommendation {:.6f}, Confidence {:.6f}'.format(ex, subj, inte, rec, conf))

			if eval_loss < best_eval_loss:
				best_eval_loss = eval_loss
				print("Saving the model!")
				dict_ = vars(args)
				dict_['model_state_dict'] = model.state_dict()
				dict_['main_state_dict'] = main_task_predictor.state_dict()
				dict_['scaffold_state_dict'] = scaffold_task_predictor.state_dict()
				dict_['criterion_state_dict'] = Criterion.state_dict()
				torch.save(dict_, os.path.join(MODEL_DIR, args.exp_name + '.pt'))
			# print("Exhaustive {}".format(list(zip(ex_preds.data, ex.data))))
			# print("Subjectivity {}".format(list(zip(subj_preds.data, subj.data))))
			# print("Intensity {}".format(list(zip(intensity_preds.data, opine.data))))
			# print("Recommendation {}".format(list(zip(rec_preds.data, recs_sc.data))))
			# print("Confidence {}".format(list(zip(conf_preds.data, confs_sc.data))))

			#logging.info('Predictions, Actual : {}'.format(str(list(zip(recs_preds_t, recs_sc_t)))))
		#break

def print_args():
	pass



def main():
	print_args()

	train()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dim', type=int, default=768,
						help='dimension of the embeddings')
	parser.add_argument('--upscale_dim', type=int, default=256,
						help='upscaled dimension of the embeddings')
	parser.add_argument('--codes', type=str, default='64,32,8',
						help='Comma separated values of the number of codes in each sequential layers')
	parser.add_argument('--batch_size', type=int, default=2,
	                    help='Batch size to run trainer.')
	# parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
	#                     help='Frequency of evaluation on the test set')
	# parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
	#                     help='Directory for storing input data')
	parser.add_argument('--learning_rate', type=float, default=0.009,
	                    help='Learning rate')
	parser.add_argument('--weight_decay', type=float, default=0.009,
	                    help='Weight decay regularizer')
	parser.add_argument('--model_dir', type=str, default='./MODELS',
	                    help='path to save the model')
	parser.add_argument('--exp_name', type=str, default='default',
	                    help='name of the experiment')

	args, unparsed = parser.parse_known_args()
	
	main()