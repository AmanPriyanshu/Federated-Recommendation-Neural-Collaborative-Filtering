import torch

class ServerNeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, item_num, predictive_factor=32):
        super(ServerNeuralCollaborativeFiltering, self).__init__()
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(4*predictive_factor, 2*predictive_factor), 
            torch.nn.ReLU(), 
            torch.nn.Linear(2*predictive_factor, predictive_factor), 
            torch.nn.ReLU(),
            torch.nn.Linear(predictive_factor, predictive_factor//2), 
            torch.nn.ReLU()
            )
        self.gmf_out = torch.nn.Linear(2*predictive_factor, 1)
        self.gmf_out.weight = torch.nn.Parameter(torch.ones(1, 2*predictive_factor))
        self.mlp_out = torch.nn.Linear(predictive_factor//2, 1)
        self.output_logits = torch.nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5           # alpha parameter, equation 13 in the paper
        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def layer_setter(self, model, model_copy):
    	for m, mc in zip(model.parameters(), model_copy.parameters()):
    		mc.data[:] = m.data[:]

    def set_weights(self, model):
    	self.layer_setter(model.mlp_item_embeddings, self.mlp_item_embeddings)
    	self.layer_setter(model.gmf_item_embeddings, self.gmf_item_embeddings)
    	self.layer_setter(model.mlp, self.mlp)
    	self.layer_setter(model.gmf_out, self.gmf_out)
    	self.layer_setter(model.mlp_out, self.mlp_out)
    	self.layer_setter(model.output_logits, self.output_logits)
    	
    def forward(self):
        return torch.tensor(0.0)

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat((self.model_blending*self.gmf_out.weight, (1-self.model_blending)*self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

if __name__ == '__main__':
    ncf = ServerNeuralCollaborativeFiltering(100, 64)
    print(ncf)