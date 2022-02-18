import torch

class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=32):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.mlp_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=2*predictive_factor)
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
        self.gmf_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=2*predictive_factor)
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

    def initialize_weights(self):
        torch.nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def forward(self, x):
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output = self.mlp_forward(user_id, item_id)
        return self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)

    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def mlp_forward(self, user_id, item_id):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)
        return self.mlp(torch.cat([user_emb, item_emb], dim=1))

    def join_output_weights(self):
        W = torch.nn.Parameter(torch.cat((self.model_blending*self.gmf_out.weight, (1-self.model_blending)*self.mlp_out.weight), dim=1))
        self.output_logits.weight = W

if __name__ == '__main__':
    ncf = NeuralCollaborativeFiltering(100, 100, 64)
    print(ncf)