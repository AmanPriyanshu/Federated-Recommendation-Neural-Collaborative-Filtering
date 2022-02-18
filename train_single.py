import torch
from dataloader import MovielensDatasetLoader
from model import NeuralCollaborativeFiltering
import numpy as np
from tqdm import tqdm
from metrics import compute_metrics
import pandas as pd

class MatrixLoader:
	def __init__(self, ui_matrix, default=None, seed=0):
		np.random.seed(seed)
		self.ui_matrix = ui_matrix
		self.positives = np.argwhere(self.ui_matrix!=0)
		self.negatives = np.argwhere(self.ui_matrix==0)
		if default is None:
			self.default = np.array([[0, 0]]), np.array([0])
		else:
			self.default = default

	def delete_indexes(self, indexes, arr="pos"):
		if arr=="pos":
			self.positives = np.delete(self.positives, indexes, 0)
		else:
			self.negatives = np.delete(self.negatives, indexes, 0)

	def get_batch(self, batch_size):
		if self.positives.shape[0]<batch_size//4 or self.negatives.shape[0]<batch_size-batch_size//4:
			return torch.tensor(self.default[0]), torch.tensor(self.default[1])
		try:
			pos_indexes = np.random.choice(self.positives.shape[0], batch_size//4)
			neg_indexes = np.random.choice(self.negatives.shape[0], batch_size - batch_size//4)
			pos = self.positives[pos_indexes]
			neg = self.negatives[neg_indexes]
			self.delete_indexes(pos_indexes, "pos")
			self.delete_indexes(neg_indexes, "neg")
			batch = np.concatenate((pos, neg), axis=0)
			if batch.shape[0]!=batch_size:
				return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()
			np.random.shuffle(batch)
			y = np.array([self.ui_matrix[i][j] for i,j in batch])
			return torch.tensor(batch), torch.tensor(y).float()
		except:
			return torch.tensor(self.default[0]), torch.tensor(self.default[1]).float()

class NCFTrainer:
	def __init__(self, ui_matrix, epochs, batch_size, latent_dim=32, device=None):
		self.ui_matrix = ui_matrix
		self.epochs = epochs
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.loader = None
		self.initialize_loader()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.ncf = NeuralCollaborativeFiltering(self.ui_matrix.shape[0], self.ui_matrix.shape[1], self.latent_dim).to(self.device)

	def initialize_loader(self):
		self.loader = MatrixLoader(self.ui_matrix)

	def train_batch(self, x, y, optimizer):
		y_ = self.ncf(x)
		mask = (y>0).float()
		loss = torch.nn.functional.mse_loss(y_*mask, y)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		return loss.item(), y_.detach()

	def train_model(self, optimizer, epochs=None, print_num=10):
		epoch = 0
		progress = {"epoch": [], "loss": [], "hit_ratio@10": [], "ndcg@10": []}
		running_loss, running_hr, running_ndcg = 0, 0, 0
		prev_running_loss, prev_running_hr, prev_running_ndcg = 0, 0, 0
		if epochs is None:
			epochs = self.epochs
		steps, prev_steps, prev_epoch = 0, 0, 0
		while epoch<epochs:
			x, y = self.loader.get_batch(self.batch_size)
			if x.shape[0]<self.batch_size:
				prev_running_loss, prev_running_hr, prev_running_ndcg = running_loss, running_hr, running_ndcg
				running_loss = 0
				running_hr = 0
				running_ndcg = 0
				prev_steps = steps
				steps = 0
				epoch += 1
				self.initialize_loader()
				x, y = self.loader.get_batch(self.batch_size)
			x, y = x.int(), y.float()
			x, y = x.to(self.device), y.to(self.device)
			loss, y_ =	self.train_batch(x, y, optimizer)
			hr, ndcg = compute_metrics(y.cpu().numpy(), y_.cpu().numpy())
			running_loss += loss
			running_hr += hr
			running_ndcg += ndcg
			if epoch!=0 and steps==0:
				results = {"epoch": prev_epoch, "loss": prev_running_loss/(prev_steps+1), "hit_ratio@10": prev_running_hr/(prev_steps+1), "ndcg@10": prev_running_ndcg/(prev_steps+1)}
			else:
				results = {"epoch": prev_epoch, "loss": running_loss/(steps+1), "hit_ratio@10": running_hr/(steps+1), "ndcg@10": running_ndcg/(steps+1)}
			steps += 1
			if prev_epoch!=epoch:
				progress["epoch"].append(results["epoch"])
				progress["loss"].append(results["loss"])
				progress["hit_ratio@10"].append(results["hit_ratio@10"])
				progress["ndcg@10"].append(results["ndcg@10"])
				prev_epoch+=1
		r_results = {"num_users": self.ui_matrix.shape[0]}
		r_results.update({i:results[i] for i in ["loss", "hit_ratio@10", "ndcg@10"]})
		return r_results, progress

	def train(self, ncf_optimizer, return_progress=False):
		self.ncf.join_output_weights()
		results, progress = self.train_model(ncf_optimizer)
		if return_progress:
			return results, progress
		else:
			return results

if __name__ == '__main__':
	dataloader = MovielensDatasetLoader()
	trainer = NCFTrainer(dataloader.ratings[:50], epochs=20, batch_size=128)
	ncf_optimizer = torch.optim.Adam(trainer.ncf.parameters(), lr=5e-4)
	_, progress = trainer.train(ncf_optimizer, return_progress=True)