# Federated-Neural-Collaborative-Filtering
Neural Collaborative Filtering (NCF) is a paper published by National University of Singapore, Columbia University, Shandong University, and Texas A&amp;M University in 2017. It utilizes the flexibility, complexity, and non-linearity of Neural Network to build a recommender system. 

Aim to federated this!

## Demo
![demo](/output.png)

## Setting:

Each client contains a group of users, in the real world this could be considered as connecting from the same WiFi. They learn a local model for recommendation, which is then aggregated centrally.

## Metrics:

1. Hit Ratio: is the fraction of users for which the correct answer is included in the recommendation list of length N, here `N=10`.
2. NDCG: is a metric of ranking quality or the relevance of the top N listed products, here `N=10`.

## Execution:

### Run the Central Single Client Model

Using the command: `python train_single.py`

```py
dataloader = MovielensDatasetLoader()
trainer = NCFTrainer(dataloader.ratings[:50], epochs=20, batch_size=128)
ncf_optimizer = torch.optim.Adam(trainer.ncf.parameters(), lr=5e-4)
_, progress = trainer.train(ncf_optimizer, return_progress=True)
```

### Run the Federated Aggregator Multi-Client Model

Using the command: `python train_federated.py`

```py
dataloader = MovielensDatasetLoader()
fncf = FederatedNCF(dataloader.ratings, num_clients=50, user_per_client_range=[1, 10], mode="ncf", aggregation_epochs=50, local_epochs=10, batch_size=128)
fncf.train()
```
