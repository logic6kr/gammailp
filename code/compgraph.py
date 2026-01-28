import torch 
import collections
import numpy as np
import pickle
TORCH_FLOAT_TYPE = torch.float32

class DkmCompGraph(torch.nn.Module):
    def __init__(self, ae_specs, n_clusters, val_lambda, input_size, device, mode='with_vae', alpha = 1):
        super(DkmCompGraph, self).__init__() 
        self.ae_specs = ae_specs
        self.input_size = input_size
        if type(ae_specs) == list:
            self.embedding_size = ae_specs[0][int((len(ae_specs[0])-1)/2)]
        else:
            self.embedding_size = ae_specs
        
        # kmeans loss computations
        ## Tensor for cluster representatives
        minval_rep, maxval_rep = -1, 1
        # cluster_rep = (maxval_rep-minval_rep) * torch.rand(n_clusters, self.embedding_size, dtype=TORCH_FLOAT_TYPE) + minval_rep
        cluster_rep = torch.tensor(np.zeros((n_clusters, self.embedding_size)), dtype=TORCH_FLOAT_TYPE, device=device)
        self.cluster_rep = torch.nn.Parameter(cluster_rep, requires_grad=True)
        
        self.n_clusters = n_clusters
        self.val_lambda = val_lambda
        self.mode = mode
        self.alpha = alpha
        self.device = device
        if self.mode == 'with_vae':
            self.encoder, self.decoder = self.build_autoencoder(ae_specs)
            self.encoder.to(device)
            self.decoder.to(device)
        
    
    def fc_layers(self, input:int, specs):
        [dimensions, activations, names] = specs
        layers = collections.OrderedDict()
        for dimension, activation, name in zip(dimensions, activations, names):
            layers[name] = torch.nn.Linear(in_features=input, out_features=dimension)
            if activation is not None:
                layers[name+'activation'] = activation
            input = dimension
        return torch.nn.Sequential(layers)
    
    def build_autoencoder(self, specs):
        [dimensions, activations, names] = specs
        mid_ind = int(len(dimensions)/2)

        # Encoder
        embedding = self.fc_layers(self.input_size, [dimensions[:mid_ind], activations[:mid_ind], names[:mid_ind]])
        # Decoder
        output = self.fc_layers(self.embedding_size, [dimensions[mid_ind:], activations[mid_ind:], names[mid_ind:]])
        
        
        return embedding, output
    
    def autoencoder(self, input):
        embedding = self.encoder(input)
        output = self.decoder(embedding)
        return embedding, output
    
    def f_func(self, x, y):
        return torch.sum(torch.square(x - y), axis=1)

    def g_func(self, x, y):
        return torch.sum(torch.square(x - y), axis=1)
    
    def get_reconstruction_loss(self, input):
        embedding, output = self.autoencoder(input)
        rec_error = self.g_func(input, output)
        ae_loss = torch.mean(rec_error)
        return ae_loss, embedding, output
    
    def forward(self, input):
        list_dist = []
        if self.mode == 'with_vae':
            self.ae_loss,embedding,_ = self.get_reconstruction_loss(input)
        else:
            embedding = input
        # for each embedding, compute the distance to each cluster representative
        
        expand_embedding = embedding.unsqueeze(1).expand(-1, self.n_clusters, -1)
        exp_cluster_rep = self.cluster_rep.unsqueeze(0).expand(embedding.shape[0], -1, -1)
        distance = torch.sum(torch.square(expand_embedding-exp_cluster_rep), axis=2)
        self.stack_dist = distance.transpose(0, 1)
        
        min_dist = torch.min(self.stack_dist, dim=0).values
        min_index = torch.argmin(self.stack_dist, dim=0)
        
        # get the centroid embeddings of the each instance
        self.centroids = self.cluster_rep[min_index, :]
        
        # get the centroid index of the each instance
        self.centroid_index = min_index
        
        min_dist_expand = min_dist.unsqueeze(0).expand(self.n_clusters, -1)
        stack_exp_batch = torch.exp(-1 * self.alpha * (self.stack_dist - min_dist_expand))
        sum_exponentials = torch.sum(stack_exp_batch, dim=0)
        
        sum_exponentials_expand = sum_exponentials.unsqueeze(0).expand(self.n_clusters, -1)
        softmax_batch = stack_exp_batch / sum_exponentials_expand
        stack_weighted_dist = softmax_batch * self.stack_dist
        
        self.stack_possibility = softmax_batch
        sum_stack_weighted_dist = torch.sum(stack_weighted_dist, dim=1)
        self.kmeans_loss = torch.mean(sum_stack_weighted_dist)
        # self.kmeans_loss = torch.mean(torch.sum(stack_weighted_dist, axis=0))
        
        if self.mode == 'with_vae':
            self.ae_loss = self.ae_loss
        else:
            self.ae_loss = 0
        
        loss = self.ae_loss + self.val_lambda *self.kmeans_loss
        
        return self.centroids, self.centroid_index, loss
    
    def get_cluster_index(self,x):
        ''''
        Do not train the weights here 
        '''
        self.eval()
        with torch.no_grad():
            centroids, center_id, loss = self.forward(x)
            return center_id