import torch
from train_single import NCFTrainer
from dataloader import MovielensDatasetLoader
import random
from tqdm import tqdm
from server_model import ServerNeuralCollaborativeFiltering

class FederatedNCF:
	def __init__(self, ui_matrix, num_clients=50, user_per_client_range=[1, 5], mode="ncf", aggregation_epochs=50, local_epochs=10, batch_size=128, latent_dim=32, seed=0):
		random.seed(seed)
		self.ui_matrix = ui_matrix
		self.num_clients = num_clients
		self.latent_dim = latent_dim
		self.user_per_client_range = user_per_client_range
		self.mode = mode
		self.aggregation_epochs = aggregation_epochs
		self.local_epochs = local_epochs
		self.batch_size = batch_size
		self.clients = self.generate_clients()
		self.ncf_optimizers = [torch.optim.Adam(client.ncf.parameters(), lr=5e-4) for client in self.clients]

	def generate_clients(self):
		start_index = 0
		clients = []
		for i in range(self.num_clients):
			users = random.randint(self.user_per_client_range[0], self.user_per_client_range[1])
			clients.append(NCFTrainer(self.ui_matrix[start_index:start_index+users], epochs=self.local_epochs, batch_size=self.batch_size))
			start_index += users
		return clients

	def single_round(self, epoch=0, first_time=False):
		single_round_results = {key:[] for key in ["num_users", "loss", "hit_ratio@10", "ndcg@10"]}
		bar = tqdm(enumerate(self.clients), total=self.num_clients)
		for client_id, client in bar:
			results = client.train(self.ncf_optimizers[client_id])
			for k,i in results.items():
				single_round_results[k].append(i)
			printing_single_round = {"epoch": epoch}
			printing_single_round.update({k:round(sum(i)/len(i), 4) for k,i in single_round_results.items()})
			model = torch.jit.script(client.ncf.to(torch.device("cpu")))
			torch.jit.save(model, "./models/local/dp"+str(client_id)+".pt")
			bar.set_description(str(printing_single_round))
		bar.close()

	def federate(self):
		for client_id in range(self.num_clients):
			model = torch.jit.load("./models/local/dp"+str(client_id)+".pt")
			item_model = ServerNeuralCollaborativeFiltering(item_num=self.ui_matrix.shape[1], predictive_factor=self.latent_dim)
			item_model.set_weights(model)
			model = torch.jit.script(model.to(torch.device("cpu")))
			torch.jit.save(model, "./models/local_items/dp"+str(client_id)+".pt")

	def train(self):
		first_time = True
		for epoch in range(self.aggregation_epochs):
			self.single_round(epoch=epoch, first_time=first_time)
			first_time = False
			self.federate()
			exit()

if __name__ == '__main__':
	dataloader = MovielensDatasetLoader()
	fncf = FederatedNCF(dataloader.ratings, num_clients=4, user_per_client_range=[1, 10], mode="ncf", aggregation_epochs=50, local_epochs=10, batch_size=128)
	fncf.train()