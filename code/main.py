import pickle, torch, itertools, datetime, argparse, datasets, tqdm, sys, os, time
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)
from ordered_set import OrderedSet
import pandas as pd
import torch.nn as nn
from torch.utils.data import  Dataset
from datasets import load_dataset, load_from_disk
from transformers import  BertTokenizer, BertForMaskedLM, BertModel
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from logic_back.dforl_torch import DeepRuleLayer
from logic_back.dforl_torch import CheckMetrics
from compgraph import DkmCompGraph
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import MNIST
import random
'''
This version implement the bert embeddings and feedforward neural network for the propositionalization layer
'''


    
class MNISTAutoencoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()

        # ---- Encoder ----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # -> 16 × 14 × 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32 × 7 × 7
            nn.ReLU(),
            nn.Flatten(),                               # -> 32*7*7 = 1568
        )

        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)  # latent vector

        # ---- Decoder ----
        self.decoder_input = nn.Linear(latent_dim, 32 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),                                   # -> 16 × 14 × 14
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),                                # -> 1 × 28 × 28
        )

    def forward(self, x):
        # encode
        enc = self.encoder(x)
        embed = self.fc_mu(enc)

        # decode
        dec_input = self.decoder_input(embed)
        dec_input = dec_input.view(-1, 32, 7, 7)
        x_hat = self.decoder(dec_input)

        return x_hat, embed
    
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------- ENCODER ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64×64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 32×32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),# 16×16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 8×8
            nn.ReLU(),
        )

        # flatten to embedding (256*8*8 = 16384 → 512)
        self.fc_enc = nn.Linear(256*14*14, 768)

        # ---------- DECODER ----------
        self.fc_dec = nn.Linear(768, 256*14*14)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32×32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 64×64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 128×128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        embeddings = self.fc_enc(x)  # the embedding
        x = self.fc_dec(embeddings)
        x = x.view(-1, 256, 14, 14)
        x = self.decoder(x)
        return x, embeddings


class FactHead(nn.Module):
    # input entities and relations, outputs the occurrence of the triples 
    def __init__(self, embedding_length):
        super(FactHead, self).__init__()
        self.checker = nn.Sequential(
                    nn.Linear(embedding_length*3, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
        # initial to zero 
        # torch.nn.init.zeros_(self.checker[0].weight)
        # torch.nn.init.zeros_(self.checker[2].weight)
        # torch.nn.init.zeros_(self.checker[0].bias)
        # torch.nn.init.zeros_(self.checker[2].bias)
        
    def forward(self, x1, x2, x3):
        #! x1 and x2 are the entity 1 and entity 2. X3 is the relation 
        # x1, x2, x3 are the embeddings of the entities and relations
        x = torch.cat([x1, x2, x3], dim=1)
        return self.checker(x)
        
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x

class NeuralPredicate(nn.Module):
    def __init__(self, output_dim = 100):
        super(NeuralPredicate, self).__init__()        
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        # set the parameter to non-trainable
        for param in self.bert.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(768, output_dim)
        self.layers = [-4, -3, -2, -1]
    def forward(self, input_ids, attention_mask, output_hidden_states):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
        x = x.hidden_states
        x = torch.stack([x[i] for i in self.layers]).sum(0)
        x = x[:,0,:]
        x = self.linear(x)
        return x

class BertHead(nn.Module):
    '''
    only a MLP of the bert output, return an embedding of the inputs 
    '''
    def __init__(self, output_dim = 100, second_dim = 10):
        super(BertHead, self).__init__()
        self.linear = nn.Linear(768, output_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, second_dim)
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x

class FactHead(nn.Module):
    '''
    only a MLP of the bert output a probability, 
    '''
    def __init__(self, output_dim = 100, second_dim = 10):
        super(FactHead, self).__init__()
        self.linear = nn.Linear(768, output_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, second_dim)
        self.last = nn.Linear(3*second_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def generate_embedding(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x
        
    
    def get_possible(self, e1, embed_r, e2):
        x = torch.cat([e1, embed_r, e2], dim=1)
        x = self.last(x)
        x = self.sigmoid(x)
        return x
    
    def forward(self, embed_e1, embed_r, embed_e2):
        # embed_e1, embed_r, embed_e2 are the embeddings of the entities and relations

        e1 = self.generate_embedding(embed_e1)
        e2 = self.generate_embedding(embed_e2)                    
        embed_r = self.generate_embedding(embed_r)
        
        x = self.get_possible(e1= e1, embed_r=embed_r, e2=e2)
        return x

class BaseRule():
    def __init__(self, ele_number, final_dim, current_dir, task_name, tokenize_length):
        self.ele_number = ele_number
        self.final_dim = final_dim
        self.current_dir = current_dir
        self.task_name = task_name
        self.writer = SummaryWriter(f'{self.current_dir}/cache/{self.task_name}/runs/')
        self.tokenize_length = tokenize_length
        
    def get_hidden_states(self, output, token_ids_word=0, layers=[-4, -3, -2, -1]):
        """Push input IDs through model. Stack and sum `layers` (last four by default).
        Select only those subword token outputs that belong to our word of interest
        and average them."""
        # Get all hidden states
        states = output.hidden_states
        # Stack and sum all requested layers
        output = torch.stack([states[i] for i in layers]).sum(0)
        # Only select the tokens that constitute the requested word
        word_tokens_output = output[:,token_ids_word,:]
        return word_tokens_output
    
    def get_distance(self, e, r, l):
        '''
        e,r,l can be tensors. The index is the key to located the entity and relation
        '''
        # get the distance between two entities
        # d(h + ℓ,t) = |h|_2^2 + |l|_2^2 + |t|_2^2 −2 hT t + ℓT (t−h)
        e_norm = torch.linalg.norm(e, ord=2, dim=1)
        r_norm = torch.linalg.norm(r, ord=2, dim=1)
        l_norm = torch.linalg.norm(l, ord=2, dim=1)
        # distance = e_norm**2 + r_norm**2 + l_norm**2 - 2 * (torch.dot(e, l) + torch.dot(r, l - e))
        distance = e_norm**2 + r_norm**2 + l_norm**2 - 2 * (torch.diagonal(torch.mm(e, torch.transpose(l,0,1)),0) + torch.diagonal(torch.mm(r, torch.transpose(l - e,0,1))))
        return distance



class BatchZ(Dataset):
    '''
    Random generate random Z objects based on all X, Y in positive examples with target predicates 
    '''
    def __init__(self, entity_path, relation_path, label_path, element_number = -1, random_negative = 20, target_atoms_index = [], target_predicate_arity = 2, number_variable = 3):
        # read tensor data
        '''
        @param
        target_atoms_index: the index of the atoms with target predicates
        '''
        if type(entity_path) == str:
            self.entity = torch.load(entity_path) 
            self.relation = torch.load(relation_path)
            self.label = torch.load(label_path)
        else:
            self.entity = entity_path
            self.relation = relation_path
            self.label = label_path
        if element_number > 0:
            self.entity = self.entity[:element_number]
            self.relation = self.relation[:element_number]
            self.label = self.label[:element_number]
        self.all_objects = torch.cat([self.entity, self.label], dim=0)
        self.selected_target_atom_index = []
        for item in target_atoms_index:
            if item < element_number or element_number < 0:
                self.selected_target_atom_index.append(item)

        self.all_objects_length = self.all_objects.shape[0]
        self.min_length = min(len(d) for d in [self.entity, self.relation, self.label])
        self.random_negative = random_negative
        # build the fact KB 
        self.all_facts = torch.concat([torch.cat([self.entity, self.relation], dim=1), self.label], dim=1)
        self.target_predicate_arity = target_predicate_arity
        self.additional_variable = number_variable - self.target_predicate_arity
        
        
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, idx):
        # get the knowledge base into GPU 
        idx_mod = idx % len(self.selected_target_atom_index)
        idx_item = self.selected_target_atom_index[idx_mod]
    
        random_z_idx = torch.randint(0,self.all_objects_length, (self.random_negative,self.additional_variable))
        idx_batch = torch.ones(self.random_negative,) * idx_item
        idx_batch = idx_batch.unsqueeze(1)
        # random_z_idx = random_z_idx.unsqueeze(1)
        # we arrange the pairs of entity into one dimension in the data
        first_entities_num = len(self.entity)
        second_entity_index = first_entities_num + idx_batch
        
        # find the random object to substitute the variable Y
        negative_second_entity_index = torch.randint(0, self.all_objects_length, (self.random_negative,))
        # no fact in the random Y
        negative_second_entity_index = torch.where(negative_second_entity_index == second_entity_index[0][0], (negative_second_entity_index + 1) % self.all_objects_length, negative_second_entity_index)
        
        # find the random object to substitute the variable X 
        negative_first_entity_index = torch.randint(0, self.all_objects_length, (self.random_negative,))
        # no fact in the random X
        negative_first_entity_index = torch.where(negative_first_entity_index == idx_batch[0][0], (negative_first_entity_index + 1) % self.all_objects_length, negative_first_entity_index)
        
        
        # connect X substitutions (correct XYR(Z), correct XYR(Z), R(X)YR(Z), XR(Y)Z)
        idx_batch_connect  = torch.cat([idx_batch, idx_batch, negative_first_entity_index.unsqueeze(1), idx_batch], dim=0)
        # connect Y substitutions 
        second_entity_index = torch.cat([second_entity_index,second_entity_index,second_entity_index, negative_second_entity_index.unsqueeze(1)], dim=0)
        # get the random objects to substitute the variable Z
        another_random_z_idx = torch.randint(0, self.all_objects_length, (self.random_negative,self.additional_variable))
        another_random_z_idx_1 = torch.randint(0, self.all_objects_length, (self.random_negative,self.additional_variable))
        another_random_z_idx_2 = torch.randint(0, self.all_objects_length, (self.random_negative,self.additional_variable))
        # combine Z substitutions 
        random_z_idx = torch.cat([random_z_idx, another_random_z_idx, another_random_z_idx_1, another_random_z_idx_2], dim=0)
        index = torch.cat([idx_batch_connect, second_entity_index, random_z_idx], dim=1)
        # combine X, Y, Z substitutions together 
        if self.target_predicate_arity == 1:
            idx_batch_connect = torch.cat([idx_batch,negative_first_entity_index.unsqueeze(1)], dim=0)
            second_entity_index = torch.cat([another_random_z_idx, another_random_z_idx_1], dim=0)
            # random_z_idx = torch.cat([another_random_z_idx_1, another_random_z_idx_2], dim=0)
            index = torch.cat([idx_batch_connect, second_entity_index], dim=1)

        return index

class BatchChain(Dataset):
    '''
    Random generate random Z objects based on all X, Y in positive examples with target predicates with a constrain of chain law
    '''
    def __init__(self, entity_path, relation_path, label_path, element_number = 10, random_negative = 20, target_atoms_index = [], target_predicate_arity = 2,index_to_entity={}, entity_to_entityindex={}, entity_to_index={}):
        # read tensor data
        '''
        @param
        target_atoms_index: the index of the atoms with target predicates
        '''
        if type(entity_path) == str:
            self.entity = torch.load(entity_path) 
            self.relation = torch.load(relation_path)
            self.label = torch.load(label_path)
        else:
            self.entity = entity_path
            self.relation = relation_path
            self.label = label_path
        if element_number > 0:
            self.entity = self.entity[:element_number]
            self.relation = self.relation[:element_number]
            self.label = self.label[:element_number]
        self.all_objects = torch.cat([self.entity, self.label], dim=0)
        self.selected_target_atom_index = []
        for item in target_atoms_index:
            if item < element_number or element_number < 0:
                self.selected_target_atom_index.append(item)

        # remove the duplicated index from target atom index 
        self.target_X_index = []
        for item in self.selected_target_atom_index:
            obj = index_to_entity[item]
            index = entity_to_index[obj][0]
            if index not in self.target_X_index:
                self.target_X_index.append(index)
        
        self.all_objects_length = self.all_objects.shape[0]
        self.min_length = min(len(d) for d in [self.entity, self.relation, self.label])
        self.random_negative = random_negative
        # build the fact KB 
        self.all_facts = torch.concat([torch.cat([self.entity, self.relation], dim=1), self.label], dim=1)
        self.target_predicate_arity = target_predicate_arity
        self.index_to_entity = index_to_entity 
        self.entity_to_entityindex = entity_to_entityindex
        self.entity_to_index = entity_to_index
        # index_to_obj =  [0] * 2 * len(self.entity)
        # index_to_obj = {}
        # for key, value in index_to_entity.items():
        #     index_to_obj[key] = value
        # self.index_to_obj = index_to_obj
        if self.target_predicate_arity == 1:
            self.target_entity = [self.index_to_entity[i] for i in self.selected_target_atom_index]
            self.target_entity_index = [entity_to_index[i][0] for i in self.target_entity]
            self.negative_entity = [i for i in self.entity_to_index.keys() if i not in self.target_entity]
            self.negative_entity_index = [entity_to_index[i][0] for i in self.negative_entity]
            
    def __len__(self):
        return self.min_length
    
    def return_representative_index(self, id):
        obj = self.index_to_entity[id]
        index = self.entity_to_index[obj][0]
        return index 
    
    def __getitem__(self, idx):
        # get the knowledge base into GPU 
        idx_mod = idx % len(self.selected_target_atom_index)
        idx_item = self.selected_target_atom_index[idx_mod]
        representative_idx = self.return_representative_index(idx_item)
        idx_batch = torch.ones(self.random_negative, dtype=torch.int64) * representative_idx
        idx_batch = idx_batch.unsqueeze(1)

        
        if self.target_predicate_arity == 2:
            idy_item = idx_item + len(self.entity)
            second_entity_index = self.return_representative_index(idy_item)
            second_entity_index = torch.ones(self.random_negative, dtype=torch.int64) * second_entity_index
            second_entity_index = second_entity_index.unsqueeze(1)
            # second_entity_index = first_entities_num + idx_batch
            # get the connected index by first entity and second entity 
            fist_entity = self.index_to_entity[idx_batch[0][0].item()]
            seocnd_entity = self.index_to_entity[second_entity_index[0][0].item()]
            connected_index_XY = (self.entity_to_entityindex[fist_entity] + self.entity_to_entityindex[seocnd_entity])
            connected_index_XY = torch.tensor([self.return_representative_index(i) for i in connected_index_XY])
            random_Z_XY_index = torch.randint(0, connected_index_XY.shape[0], (self.random_negative,))
            random_Z_XY = connected_index_XY[random_Z_XY_index].unsqueeze(1)
            random_Z_XY_index_1 = torch.randint(0, connected_index_XY.shape[0], (self.random_negative,))
            random_Z_XY_1 = connected_index_XY[random_Z_XY_index_1].unsqueeze(1)
            
            

            # find the random object to substitute the variable X 
            negative_first_entity_index = np.random.randint(0, self.all_objects_length, (self.random_negative,))
            representative_negative_first_entity_index = [self.return_representative_index(i) for i in negative_first_entity_index]
            representative_negative_first_entity_index = torch.tensor(representative_negative_first_entity_index).unsqueeze(1)
            # no fact in the random X
            # negative_first_entity_index = torch.where(negative_first_entity_index == idx_batch[0][0], (negative_first_entity_index + 1) % self.all_objects_length, negative_first_entity_index).unsqueeze(1)
            
            # find the random object to substitute the variable Y
            negative_second_entity_index = np.random.randint(0, self.all_objects_length, (self.random_negative,))
            # no fact in the random Y
            # negative_second_entity_index = torch.where(negative_second_entity_index == second_entity_index[0][0], (negative_second_entity_index + 1) % self.all_objects_length, negative_second_entity_index).unsqueeze(1)
            representative_negative_second_entity_index = [self.return_representative_index(i) for i in negative_second_entity_index]
            representative_negative_second_entity_index = torch.tensor(representative_negative_second_entity_index).unsqueeze(1)
            
            # connect X substitutions (correct X Y R_conXY(Z), correct X Y R_conXY(Z), R_disconY(X)Y R_con(XYZ), XR_disconX(Y)R_conXY(Z))
            idx_batch_connect  = torch.cat([idx_batch, idx_batch, representative_negative_first_entity_index, idx_batch], dim=0)
            
            # connect Y substitutions 
            second_entity_index = torch.cat([second_entity_index,second_entity_index,second_entity_index, representative_negative_second_entity_index], dim=0)

            # get the random objects to substitute the variable Z
            another_random_z_idx_1 = np.random.randint(0, self.all_objects_length, (self.random_negative,))
            another_random_z_idx_1 = [self.return_representative_index(i) for i in another_random_z_idx_1]
            another_random_z_idx_1 = torch.tensor(another_random_z_idx_1).unsqueeze(1)
            
            another_random_z_idx_2 = np.random.randint(0, self.all_objects_length, (self.random_negative,))
            another_random_z_idx_2 = [self.return_representative_index(i) for i in another_random_z_idx_2]
            another_random_z_idx_2 = torch.tensor(another_random_z_idx_2).unsqueeze(1)
            
            
            random_z_idx = torch.cat([random_Z_XY, random_Z_XY_1, another_random_z_idx_1, another_random_z_idx_2], dim=0)
            
            # combine X, Y, Z substitutions together
            index = torch.cat([idx_batch_connect, second_entity_index, random_z_idx], dim=1)
        elif self.target_predicate_arity == 1:
            # todo: The substitutation are generated based on the chain law
            # connect X substitutions (correct X R_conX(Y) R_conXY(Z),  R(X)R(Y)R(Z))
            # get the connected index by first entity and second entity 
            fist_entity = self.index_to_entity[idx_batch[0][0].item()]
            connected_index_X = self.entity_to_entityindex[fist_entity]
            Y_obj = list(set([self.index_to_entity[i] for i in connected_index_X]))
            connected_index_X = torch.tensor([self.entity_to_index[i][0] for i in Y_obj])
            
            random_Y_cX_index = torch.randint(0, connected_index_X.shape[0], (self.random_negative,))
            random_Y_cx = connected_index_X[random_Y_cX_index].unsqueeze(1)
            random_Y_cx_list = random_Y_cx.reshape(-1).numpy().tolist()
                
            # get all index Y 
            possible_Y = list(set([self.index_to_entity[i] for i in random_Y_cx_list]))
            # possible_Y = self.index_to_obj[random_Y_cx].reshape(-1).numpy().tolist()
            
            connected_z_tensors = [self.entity_to_entityindex[k] for k in possible_Y if k in self.entity_to_entityindex]
            possible_Z_pool = []
            for tensor in connected_z_tensors:
                possible_Z_pool.extend(tensor)
            # possible z obj  
            z_obj  = list(set([self.index_to_entity[i] for i in possible_Z_pool]))
            possible_Z_pool = np.array([self.entity_to_index[i][0] for i in z_obj])
            random_Z = torch.randint(0, len(possible_Z_pool), (self.random_negative,))
            possible_Z_pool = torch.tensor(possible_Z_pool)
            random_z_index = possible_Z_pool[random_Z].unsqueeze(1)
            
            # negative x 
            negative_x = torch.randint(0, len(self.negative_entity_index) , (1,))
            self.negative_entity_index = torch.tensor(self.negative_entity_index)
            negative_index = torch.tensor(self.negative_entity_index[negative_x])
            negative_index = torch.ones_like(idx_batch) * negative_index.item()
            
            neg_fist_entity = self.index_to_entity[negative_index[0][0].item()]
            neg_connected_index_X = self.entity_to_entityindex[neg_fist_entity]
            neg_Y_obj = list(set([self.index_to_entity[i] for i in neg_connected_index_X]))
            neg_connected_index_X = torch.tensor([self.entity_to_index[i][0] for i in neg_Y_obj])
            
            neg_random_Y_cX_index = torch.randint(0, neg_connected_index_X.size(0), (self.random_negative,))
            neg_random_Y_cx = neg_connected_index_X[neg_random_Y_cX_index].unsqueeze(1)
            neg_random_Y_cx_list = neg_random_Y_cx.reshape(-1).numpy().tolist()
            
            
            # get all index Y 
            neg_possible_Y = list(set([self.index_to_entity[i] for i in neg_random_Y_cx_list]))            
            neg_connected_z_tensors = [self.entity_to_entityindex[k] for k in neg_possible_Y if k in self.entity_to_entityindex]
            neg_possible_Z_pool = []
            for tensor in neg_connected_z_tensors:
                neg_possible_Z_pool.extend(tensor)
            # possible z obj  
            neg_z_obj  = list(set([self.index_to_entity[i] for i in neg_possible_Z_pool]))
            neg_possible_Z_pool = np.array([self.entity_to_index[i][0] for i in neg_z_obj])
            neg_random_Z = torch.randint(0, len(neg_possible_Z_pool), (self.random_negative,))
            neg_possible_Z_pool = torch.tensor(neg_possible_Z_pool)
            neg_random_z_index = neg_possible_Z_pool[neg_random_Z].unsqueeze(1)
            
            
            idx_batch_connect = torch.cat([idx_batch,negative_index], dim=0)
            second_entity_index = torch.cat([random_Y_cx, neg_random_Y_cx], dim=0)
            random_z_idx = torch.cat([random_z_index, neg_random_z_index], dim=0)
            index = torch.cat([idx_batch_connect, second_entity_index, random_z_idx], dim=1)
        return index

class Batch_PI_auto_encoder(Dataset):
    '''
    predicate invention for learning from images 
    Random generate random Z objects based on all X, Y in positive examples with target predicates 
    '''
    def __init__(self, train_positive, train_negative, number_variable = 3, inner_batch = 2):
        '''
        @param
        target_atoms_index: the index of the atoms with target predicates
        '''
        self.train_positive = train_positive
        self.train_negative = train_negative
        self.number_variable = number_variable # ! this should be the number of objects in the single image
        self.object_number = 3 #! this should be the number of objects in the single image or just 2 with inner batch 
        self.number_positive = int(len(train_positive) /4)
        self.number_negative = int(len(train_negative)/4)
        self.inner_batch = inner_batch
        
        
    def __len__(self):
        return self.number_positive + self.number_negative 
    
    def __getitem__(self, idx):
        if idx < self.number_positive:
            # get the positive example, each example with 4 continues images
            data = torch.stack([self.train_positive[idx*4 + i][0] for i in range(4)])
            # data = self.train_positive[idx]
            label = 1
        else:
            # get the negative example with 4 continues images and return to a subbatch, self.train_negative is a dataset already 
            data = torch.stack([self.train_negative[(idx - self.number_positive)*4 + i][0] for i in range(4)])
            # data = self.train_negative[idx - self.number_positive]
            label = 0 
            
        # get the variable 
        # data_length = len(data)
        # index = torch.randint(0, data_length, (self.inner_batch, self.object_number)) # get the object number or 2 with self.inner_batch
        
        # assign the variable to the data
        # x = data[index]
        
        return data, label    


class Batch_PI(Dataset):
    '''
    predicate invention for learning from images 
    Random generate random Z objects based on all X, Y in positive examples with target predicates 
    '''
    def __init__(self, train_positive, train_negative, number_variable = 3, inner_batch = 2):
        '''
        @param
        target_atoms_index: the index of the atoms with target predicates
        '''
        self.train_positive = train_positive
        self.train_negative = train_negative
        self.number_variable = number_variable # ! this should be the number of objects in the single image
        self.object_number = 3 #! this should be the number of objects in the single image or just 2 with inner batch 
        self.number_positive = len(train_positive)
        self.number_negative = len(train_negative)
        self.inner_batch = inner_batch
        
    def __len__(self):
        return self.number_positive + self.number_negative 
    
    def __getitem__(self, idx):
        if idx < self.number_positive:
            # get the positive example 
            data = self.train_positive[idx]
            label = 1
        else:
            # get the negative example 
            data = self.train_negative[idx - self.number_positive]
            label = 0 
            
        # get the variable 
        # data_length = len(data)
        # index = torch.randint(0, data_length, (self.inner_batch, self.object_number)) # get the object number or 2 with self.inner_batch
        
        # assign the variable to the data
        # x = data[index]
        
        return data, label


class Batch_PartialPI(BatchZ):
    '''
    predicate invention for learning from images 
    Random generate random Z objects based on all X, Y in positive examples with target predicates 
    '''
    def __init__(self, train_positive, number_variable = 3, inner_batch = 2, entity_path=None, relation_path=None, label_path=None, random_negative = 20, target_predicate_arity = 2,target_atoms_index=None):
        '''
        @param
        target_atoms_index: the index of the atoms with target predicates
        '''
        # initial fater class BatchZ
        super().__init__(entity_path, relation_path, label_path, element_number=-1, random_negative=random_negative, target_atoms_index=target_atoms_index, target_predicate_arity=target_predicate_arity,number_variable=number_variable)

        self.train_positive = train_positive.unsqueeze(0) 
        # self.train_negative = train_negative
        self.number_variable = number_variable # ! this should be the number of objects in the single image
        self.object_number = 3 #! this should be the number of objects in the single image or just 2 with inner batch 
        self.number_positive = len(self.train_positive)
        # self.number_negative = len(train_negative)
        self.inner_batch = inner_batch

    
    def __getitem__(self, idx):
        fact_index = super().__getitem__(idx)

        # only one positive 
        # if idx < self.number_positive:
        #     # get the positive example 
        #     data = self.train_positive[idx]
        #     label = 1
        # else:
        #     # get the negative example 
        #     data = self.train_negative[idx - self.number_positive]
        #     label = 0 
        data = self.train_positive[0]
            
        # get the variable 
        data_length = len(data)
        index = torch.randint(0, data_length, (self.inner_batch, self.object_number)) # get the object number or 2 with self.inner_batch
        
        # assign the variable to the data
        image_facts = data[index]
        
        return image_facts, fact_index

class AllSubstitution(Dataset):
    '''
    Random generate random Z objects based on all X, Y in positive examples with target predicates with a constrain of chain law
    '''
    def __init__(self, entity_path, relation_path, label_path, element_number = 10, random_negative = 20, target_atoms_index = [], target_predicate_arity = 2,index_to_entity={}, entity_to_index={}):
        # read tensor data
        '''
        @param
        target_atoms_index: the index of the atoms with target predicates
        '''
        if type(entity_path) == str:
            self.entity = torch.load(entity_path) 
            self.relation = torch.load(relation_path)
            self.label = torch.load(label_path)
        else:
            self.entity = entity_path
            self.relation = relation_path
            self.label = label_path
        if element_number > 0:
            self.entity = self.entity[:element_number]
            self.relation = self.relation[:element_number]
            self.label = self.label[:element_number]
        self.all_objects = torch.cat([self.entity, self.label], dim=0)
        self.selected_target_atom_index = []
        for item in target_atoms_index:
            if item < element_number or element_number < 0:
                self.selected_target_atom_index.append(item)

        self.all_objects_length = self.all_objects.shape[0]
        self.min_length = min(len(d) for d in [self.entity, self.relation, self.label])
        self.random_negative = random_negative
        # build the fact KB 
        self.all_facts = torch.concat([torch.cat([self.entity, self.relation], dim=1), self.label], dim=1)
        self.target_predicate_arity = target_predicate_arity
        self.index_to_entity = index_to_entity 
        self.entity_to_index = entity_to_index
        
        index_to_obj =  [0] * 2 * len(self.entity)
        for key, value in index_to_entity.items():
            index_to_obj[key] = value
        self.index_to_obj = (index_to_obj)
        # get all index 
        if self.target_predicate_arity == 2:
            all_index_x = self.selected_target_atom_index
            x_obj = list(set([self.index_to_obj[i] for i in all_index_x]))
            all_index_X_nod = [self.entity_to_index[i][0] for i in x_obj]
            
            all_index_y = [i+len(self.entity) for i in all_index_x]
            all_index_y_obj = list(set([self.index_to_obj[i] for i in all_index_y]))
            all_inedx_y_nod = [self.entity_to_index[i][0] for i in all_index_y_obj]
            
            all_index_z = [value[0] for value in self.entity_to_index.values()]
            
            self.all_substitution = list(itertools.product(all_index_X_nod, all_inedx_y_nod, all_index_z))
        elif self.target_predicate_arity == 1:
            positive_target_x = self.selected_target_atom_index
            target_x_obj = [self.index_to_obj[i] for i in positive_target_x]
            all_target_x_index = []
            for i in target_x_obj:
                current_index = self.entity_to_index[i][0]
                all_target_x_index.append(current_index)
            all_index_y = [value[0] for value in self.entity_to_index.values()]
            all_index_z = [value[0] for value in self.entity_to_index.values()]
            negative_target_x = []
            for key in self.entity_to_index.keys():
                if key not in target_x_obj:
                    negative_target_x.append(self.entity_to_index[key][0])
            negative_target_x = list(set(negative_target_x))
            all_index_x = all_target_x_index + negative_target_x
            self.all_substitution = list(itertools.product(all_index_x, all_index_y, all_index_z))

        
        
    def __len__(self):
        return len(self.all_substitution)
    
    def __getitem__(self, idx):
        # get the knowledge base into GPU 
        s_X = self.all_substitution[idx][0]
        s_Y = self.all_substitution[idx][1]
        s_Z = self.all_substitution[idx][2]
        index = (s_X, s_Y, s_Z)
        # obtain X,Y,X into three dimensional 
        return index



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, early_stop=5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')


    def early_stop(self, validation_loss):
        if self.patience == False:
            return False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter = self.counter + 1
            if self.counter >= self.patience:
                return True
        return False


class TextTwoMNISTDataset(Dataset):
    """
    Dataset that:
    - Precomputes BERT embeddings of all relation texts in __init__
    - Provides random MNIST images for given labels
    """

    def __init__(self, metadata_file, device, root="./mnist_data",
                 img_transform=None, bert_name="bert-base-uncased",
                 train=True, download=True):

        self.device = device

        # ---- 1. Load MNIST dataset ----
        self.mnist = MNIST(root=root, train=train, download=download)

        # group MNIST images by label
        self.images_by_label = {i: [] for i in range(10)}
        for img, label in self.mnist:
            self.images_by_label[label].append(img)

        # ---- 2. Load relation text and labels from metadata ----
        self.records = []
        self.texts = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                l1, rel, l2 = line.split(" ")
                self.records.append((int(l1), rel, int(l2)))
                self.texts.append(rel)

        # ---- 3. Load BERT and compute all embeddings only once ----
        tokenizer = BertTokenizer.from_pretrained(bert_name)
        bert = BertModel.from_pretrained(bert_name).to(device)
        bert.eval()

        self.text_embeddings = []

        with torch.no_grad():
            for text in self.texts:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=32
                ).to(device)

                outputs = bert(**inputs)
                cls = outputs.last_hidden_state[:, 0, :].squeeze(0)  # (hidden_size,)
                self.text_embeddings.append(cls.cpu())   # store on CPU to save GPU RAM
                
        # ---- 4. Preselect MNIST images for each record ----
        print("Preselecting MNIST images...")
        self.img1_list = []
        self.img2_list = []
        for l1, _, l2 in self.records:
            self.img1_list.append(random.choice(self.images_by_label[l1]))
            self.img2_list.append(random.choice(self.images_by_label[l2]))

        print(f"Dataset ready: {len(self.records)} samples.")

        self.img_transform = img_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # already precomputed
        text_embed = self.text_embeddings[idx]
        img1 = self.img1_list[idx]
        img2 = self.img2_list[idx]
        l1,_, l2 = self.records[idx]

        if self.img_transform:
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)

        return {
            "text_embed": text_embed,  # tensor (hidden_size,)
            "img1": img1,
            "img2": img2,
            "label1": l1,
            "label2": l2,
        }
    

class MultiFolderDataset(Dataset):
    def __init__(self, parent_dir, transform=None, exclude_name="result_preview.png"):
        self.transform = transform
        parent = Path(parent_dir)

        self.img_paths = []

        # loop through all subfolders
        for sub in parent.iterdir():
            if not sub.is_dir():
                continue

            # collect files inside this subfolder (excluding a.png)
            files = [p for p in sub.iterdir() if p.is_file() and p.name != exclude_name]

            # only keep this subfolder if it has exactly 5 files
            if len(files) != 4:
                continue

            # add files to dataset list
            self.img_paths.extend(files)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, str(img_path)
    


class DifferentiablePropositionalization(nn.Module, BaseRule):
    def __init__(self, entity_path, relation_path, epoch=10, batch_size =512, early_stop = 5, ele_number=10, current_dir='',device='', task_name='',final_dim = 10,threshold = 1500, train_loop_logic=10, model_name_specific = None, tokenize_length=10, load_pretrain_head=True, lr_predicate_model = 1e-4, lr_rule_model = 1e-3, lr_dkm  = 1e-4, target_predicate = 'Host a visit', data_format = 'kg', cluster_numbers = 10, alpha=1, number_variable=3, target_variable_arrange = 'X@Y', stop_recall = 0.97, substitution_method = 'random', random_negative = 32, open_neural_predicate = False, output_file_name = '', minimal_precision = 0.5, body_arity = '',lambda_dkm = 1,pre_train_ae_num_epochs=30):
        super(DifferentiablePropositionalization, self).__init__()
        BaseRule.__init__(self, ele_number, final_dim, current_dir, task_name,tokenize_length)
        # import bert and token 
        self.body_arity = body_arity
        self.data_format =  data_format
        self.entity_path = entity_path
        self.relation_path = relation_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.early_stop = EarlyStopper(patience=early_stop, min_delta=0.1)
        self.target_relation = target_predicate
        if target_variable_arrange == 'X@Y':
            self.target_arity = 2
            target_variable_arrange_index = (0,1)
        elif target_variable_arrange == 'X@X':
            self.target_arity = 1
            target_variable_arrange_index = (0,0)
        self.device = device
        self.substitution_method = substitution_method
        self.random_negative = random_negative
        self.bert_dim_length = 768
        self.lr_rule_model = lr_rule_model
        self.lr_predicate_model = lr_predicate_model
        self.lr_dkm  = lr_dkm
        self.threshold = threshold
        self.train_loop_logic = train_loop_logic
        self.minimal_precision = minimal_precision
        self.open_neural_predicate = open_neural_predicate
        self.open_autoencoder = False
        self.rule_path = f"{self.current_dir}/cache/{self.task_name}/rule_output_{self.target_relation}_{output_file_name}.md"
        self.writer = SummaryWriter(f'{self.current_dir}/cache/{self.task_name}/runs/')
        if model_name_specific == None:
            self.datetime = datetime.datetime.now().strftime("%d%H")
        else: 
            self.datetime = model_name_specific
        if not os.path.exists(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}"):
            os.makedirs(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}")
        self.model_path_predicate_path = f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/bert_fine_tune.pt"
        self.output_model_path_basic = f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/soft_proposition"
        self.mapping = {0:'X', 1:'Y', 2:'Z', 3:'W', 4:'V', 5:'U', 6:'T', 7:'S', 8:'R', 9:'Q'}
        self.left_bracket = '['
        self.right_bracket = ']'
        self.split = '@'
        self.end = '#'
        self.lambda_dkm = lambda_dkm
        self.pre_train_ae_num_epochs = pre_train_ae_num_epochs
        
        ## TODO user defined based on the task
        self.number_variable = number_variable
        self.variable_perturbations_list = list(itertools.permutations(range(self.number_variable), 2)) + [(i,i) for i in range(self.number_variable)]
        # [(0,0), (1,1), (2,2)]
        self.get_index_X_Y = self.variable_perturbations_list.index(target_variable_arrange_index)
        self.target_variable_arrange = target_variable_arrange
        self.stop_recall = stop_recall
        
        # end for user defined 
        self.number_variable_terms  = len(self.variable_perturbations_list)
        self.variable_perturbations = torch.tensor(self.variable_perturbations_list, device=self.device)
        self.load_pretrain_head = load_pretrain_head
        
        # write the pertumation to the file
        with open(f"{self.current_dir}/cache/{self.task_name}/variable_perturbations.txt", "w") as f:
            f.write(str(self.variable_perturbations))
            f.close()
        
        if self.data_format == 'image' or self.data_format == 'pi' or self.data_format == 'ppi' or self.data_format == 'pi_ae' or self.data_format == 'image_ae':
            self.n_clusters = cluster_numbers
            self.alpha = alpha

    def make_tokeniz(self, data_type='train'):
        '''
        # ! Run this function in the first time 
        '''
        # read train data and generated entity and relation
        
        train_data = load_dataset('csv', data_files = f"{self.current_dir}/cache/{self.task_name}/{data_type}_embeddings.csv")['train']

        if self.ele_number > 0:
            train_data = train_data.select(range(self.ele_number))
        
        # build the target knowledge data for validation
        if self.data_format == 'kg':
            train_data_list = list(train_data)
            self.build_target_validation_data(train_data_list)
        
        # entity_data = {'text':OrderedSet([])}
        relation_data = {'text':OrderedSet([])}
        
        for item in train_data:
            # entity_data['text'].add(item['textE'])
            relation_data['text'].add(item['textR'])
        # entity_data['text'] = list(entity_data['text'])
        relation_data['text'] = list(relation_data['text'])
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenize_function(examples):
            token =  self.tokenizer(examples["text"], return_tensors="pt", padding='max_length', truncation=True, max_length=self.tokenize_length)
            return token
        # entity_data = datasets.Dataset.from_dict(entity_data)
        relation_data = datasets.Dataset.from_dict(relation_data)
        # entity_data = entity_data.map(tokenize_function, batched=True)
        relation_data = relation_data.map(tokenize_function, batched=True)
        # save them to disk
        # entity_data.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/entity_token.hf")
        relation_data.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/relation_token.hf")
    
    def check_arity_predicate(self, train_data):
        # get all predicate 
        all_predicate = []
        for item in train_data:
            if item['textR'] not in all_predicate:
                all_predicate.append(item['textR'])
        # check the arity of the predicate
        predicate_arity = {}
        for i in all_predicate:
            predicate_arity[i] = 1
        for item in train_data:
            if item['text1'] != item['text2']:
                if item['textR'] in predicate_arity:
                    predicate_arity[item['textR']] = 2
                else:
                    raise('The predicate is not in the list')
        return predicate_arity
        
    def get_connected_entity(self,data):
        # build dic for all object index
        entity_to_entityindex = defaultdict(list)
        entity_to_index = defaultdict(list) 
        all_number_entity = len(data)
        index_to_entity = {}
        ini_index = 0
        for item in data:
            index_to_entity[ini_index] = item['text1']
            index_to_entity[ini_index + all_number_entity] = item['text2']
            first_entity = item['text1']
            second_entity = item['text2']
            # the connected entities 
            entity_to_entityindex[first_entity].append(ini_index+all_number_entity)
            entity_to_index[first_entity].append(ini_index)
            entity_to_entityindex[second_entity].append(ini_index)
            entity_to_index[second_entity].append(ini_index + all_number_entity)
            ini_index += 1
        return entity_to_entityindex, index_to_entity, entity_to_index

    def get_connected_entity_cluster(self, cluster_labels, single_object_numbers):
        # build dic for all object index
        # cluster labels for all objects
        entity_to_entityindex = defaultdict(list)
        entity_to_index = defaultdict(list)
        text1 = [i for i in cluster_labels[:single_object_numbers]]
        text2 = [i for i in cluster_labels[single_object_numbers:]]
        all_number_entity = (single_object_numbers)
        index_to_entity = {}
        ini_index = 0
        for item in range(all_number_entity):
            first_entity = text1[item]
            second_entity = text2[item]
            index_to_entity[ini_index] = first_entity
            index_to_entity[ini_index + single_object_numbers] = second_entity
            # the connected entities 
            entity_to_entityindex[first_entity].append(ini_index+all_number_entity)
            entity_to_index[first_entity].append(ini_index)
            entity_to_entityindex[second_entity].append(ini_index)
            entity_to_index[second_entity].append(ini_index + all_number_entity)
            ini_index += 1
        return entity_to_entityindex, index_to_entity, entity_to_index
        
    def return_representative_index(self, id):
        obj = self.index_to_entity[id]
        index = self.entity_index[obj][0]
        return index 
    
    def update_all_fact_embedding_image(self, all_object_embeddings, selected_train_R):
        number_entity = len(selected_train_R)
        embedding_e1 = all_object_embeddings[:number_entity]
        embedding_e2 = all_object_embeddings[number_entity:2*number_entity]
        all_fact_embedding = torch.cat([embedding_e1, embedding_e2, selected_train_R], dim=1)
        return all_fact_embedding


    def batched_row_existence(self, triples, all_facts, batch_size=128):
        """
        Returns a boolean tensor of shape [triples.size(0)], where each element is True
        if the corresponding row in `triples` exists in `all_facts`.

        Args:
            triples (Tensor): [N1, d] tensor
            all_facts (Tensor): [N2, d] tensor
            batch_size (int): size of batches to process at a time

        Returns:
            Tensor: [N1] boolean tensor
        """
        n = triples.size(0)
        exists = []

        for i in range(0, n, batch_size):
            batch = triples[i:i+batch_size]  # [B, d]
            # Compare batch to all_facts: [B, 1, d] vs [1, N2, d] → [B, N2, d]
            # labels_body = torch.isclose(batch, all_facts, rtol=1e-5, atol=1e-8).all(dim=2).any(1)
            labels_body = (batch==all_facts).all(dim=2).any(dim=1)
            exists.append(labels_body)

        return torch.cat(exists, dim=0)

    def rule_body_train(self):
        '''
        Read all entities and process and propositionalalization and rule learning task together.
        Train distance and logic model together.
        BERT embedding is fix but head embeddings is trainable.
        '''
        # read all entities 
        if self.data_format == 'kg':
            train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/train_embeddings.csv")['train']
        elif self.data_format == 'image':
            train_data = load_dataset('csv',  data_files=f"{self.current_dir}/cache/{self.task_name}/train_embeddings.csv")['train']
            # get all text 1
            if self.ele_number > 0:
                all_label1 = [i for i in train_data['text1'][:self.ele_number]]
                all_label2 = [ i for i in train_data['text2'][:self.ele_number]]
            else:
                all_label1 = [i for i in train_data['text1']]
                all_label2 = [ i for i in train_data['text2']]
            self.all_entity_labels = all_label1 + all_label2
        else:
            raise('The data format is wrong')
        all_relations_in_order = [i for i in train_data['textR']]
        # get the entity to entity_index list 
        self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity(train_data)
        
        
        # get the relations  occurrence index
        all_relation_occurrence_index = {}
        unique_relations  = 0
        for i in all_relations_in_order:
            if i not in all_relation_occurrence_index:
                all_relation_occurrence_index[i] = unique_relations
                unique_relations += 1

        self.all_relation_index = torch.tensor([i for i in range(len(all_relations_in_order))]).to(self.device)
        
        # get the unique relation and use the first relation index as the index 
        self.all_relation_mapping = []
        for i in train_data['textR']:
            self.all_relation_mapping.append(all_relation_occurrence_index[i])
        self.all_relation_mapping = torch.tensor(self.all_relation_mapping).to(self.device)
        
        
        self.all_first_entity = [self.return_representative_index(i) for i in range(len(train_data['text1']))]
        self.all_first_entity = torch.tensor(self.all_first_entity).unsqueeze(0).to(self.device)
        
        self.all_second_entity = [self.return_representative_index(i) for i in range(len(train_data['text1']), 2*len(train_data['text1']) )]
        self.all_second_entity = torch.tensor(self.all_second_entity).unsqueeze(0).to(self.device)

        self.all_relation_index = self.all_relation_mapping[self.all_relation_index].unsqueeze(0)
        self.all_fact_index = torch.concat([self.all_first_entity, self.all_second_entity, self.all_relation_index], dim=0).T
        
        self.unique_relation_index = torch.unique(self.all_relation_index, sorted=False).unsqueeze(0)
        self.unique_relation_index.expand(self.number_variable_terms, -1)
        self.unique_relation_index = self.unique_relation_index.unsqueeze(2)
        
        relation_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/relation_token.hf")
        target_atoms_index = []
        ini_index = 0
        for item in train_data:
            if item['textR'] == self.target_relation:
                target_atoms_index.append(ini_index)
            ini_index = ini_index + 1
        self.tar_predicate_index = None
        number_relations = unique_relations
        self.all_relations = relation_data['text']
        print('All relations:', self.all_relations) 

        # get predicate arity 
        predicate_arity = self.check_arity_predicate(train_data)
        print('Predicate arity:', predicate_arity)


        
        # get the arity of the target predicate 
        self.target_predicate_arity = predicate_arity[self.target_relation]
        
        # get all atom arrangement 
        self.all_unground_atoms = []
        for arrage in self.variable_perturbations_list:
            for item in self.all_relations:
                self.all_unground_atoms.append(f"{item}[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")
        target_index = self.all_unground_atoms.index(f'{self.target_relation}[{self.target_variable_arrange}]')
        for item in range(number_relations):
            if relation_data[item]['text'] == self.target_relation:
                self.tar_predicate_index = item
                break
        self.target_atom_index = number_relations * self.get_index_X_Y + self.tar_predicate_index
        # self.target_atom_index = self.get_index_X_Y * self.number_variable_terms + self.target_atom_index
        assert self.target_atom_index == target_index, f"Target atom index {self.target_atom_index} is not equal to the target index {target_index}"
        
        # get the valid list and valid unground atoms 
        invalid_atom_index = []
        valid_atom_index = []
        for ele in range(len(self.all_unground_atoms)):
            item = self.all_unground_atoms[ele]
            predicate = item.split('[')[0]
            first_variable = item.split('[')[1].split('@')[0]
            second_variable = item.split('[')[1].split('@')[1].split(']')[0]
            if first_variable == second_variable:
                atom_variable = 1
            else:
                atom_variable = 2
            if (predicate_arity[predicate] == 2 and atom_variable == 1) or (predicate_arity[predicate] == 1 and atom_variable == 2):
                invalid_atom_index.append(ele)
            else:
                valid_atom_index.append(ele)
        self.valid_atom_index = valid_atom_index
        self.invalid_atom_index = invalid_atom_index
        self.valid_unground_atoms = []
        for ele in valid_atom_index:
            self.valid_unground_atoms.append(self.all_unground_atoms[ele])
        print('Valid unground atoms:', self.valid_unground_atoms)
        
        # ! find and update the target atom index in the valid unground atoms
        self.target_atom_index = self.valid_unground_atoms.index(f'{self.target_relation}[{self.target_variable_arrange}]')
        
        
        # load the positive and negative data and an random index 
        train_data_E = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_E_{self.tokenize_length}.pt")
        train_data_R = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_R_{self.tokenize_length}.pt")
        train_data_L = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_L_{self.tokenize_length}.pt")

        if self.ele_number > 0:    
            selected_train_E = train_data_E[:self.ele_number].to(self.device)
            selected_train_L = train_data_L[:self.ele_number].to(self.device)
            selected_train_R = train_data_R[:self.ele_number].to(self.device)
        else:
            selected_train_E = train_data_E.to(self.device)
            selected_train_L = train_data_L.to(self.device)
            selected_train_R = train_data_R.to(self.device)
        
        # prepare build all KB with bert embedding 
        #! the arrange of embeddings is Entity1 Entity2 Relation
        all_facts_bert_embeddings = torch.cat([selected_train_E, selected_train_L, selected_train_R], dim=1) # get all true index of fact
        
        all_obj_bert_embeddings = torch.cat([selected_train_E, selected_train_L], dim=0) # get all true index of object
        
        # Unique the relation bert embeddings and keep the first occurrence as the index 
        # Step 1: Track unique rows and map them to indices
        seen = {}
        unique_rows = []
        inverse_indices = []

        for row in selected_train_R:
            row_tuple = tuple(row.tolist())
            if row_tuple not in seen:
                seen[row_tuple] = len(seen)  # assign new index
                unique_rows.append(row)
            inverse_indices.append(seen[row_tuple])

        # Step 2: Convert to tensors
        unique_tensor = torch.stack(unique_rows)               # (num_unique, D)
        inverse_tensor = torch.tensor(inverse_indices)         # (N,)

        print("Original tensor:\n", selected_train_R)
        print("Unique rows (first occurrence order):\n", unique_tensor)
        print("Inverse indices:\n", inverse_tensor)
        
        # all_relation_bert_embedding [44,768]
        selected_r_bert_embeddings = unique_tensor.unsqueeze(0)
        original_shape_selected_r_bert_embeddings = unique_tensor
        selected_r_bert_embeddings  = selected_r_bert_embeddings.expand(self.number_variable_terms, -1, -1)
        selected_r_bert_embeddings = selected_r_bert_embeddings.unsqueeze(2)
        
        
        # define the data 
        if self.data_format == 'kg':
            if self.substitution_method == 'random' or self.number_variable > 3:
                tuple_data = BatchZ(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity, number_variable = self.number_variable)
                substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True)
            elif self.substitution_method == 'chain_random':
                tuple_data = BatchChain(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity,index_to_entity=self.index_to_entity, entity_to_entityindex=self.entity_entityindex, entity_to_index = self.entity_index)
                substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=self.batch_size, shuffle=True)
            elif self.substitution_method == 'all':
                tuple_data = AllSubstitution(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity,index_to_entity=self.index_to_entity, entity_to_index=self.entity_index)
                substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=self.random_negative, shuffle=True)
        # load pretrain weights 
        head_model = FactHead(200)
        #! load pretrain weights
        if self.load_pretrain_head:
            checkpoint = torch.load(self.model_path_predicate_path, map_location='cpu')
            head_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            pass
        head_model.to(self.device)

        # Load neural logic rule learning model 
        self.logic_model = DeepRuleLayer(in_size=len(self.valid_unground_atoms)-1, rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=self.rule_path, t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=self.target_arity,time_stamp='')
        self.logic_model.to(self.device)
        
        # Define the differentiable k-means module for image task
        if self.data_format == 'image':
            self.cluster = DkmCompGraph(ae_specs=self.bert_dim_length, n_clusters=self.n_clusters, val_lambda=1, input_size=None, alpha=self.alpha,device=self.device, mode='without_vae')
            # with all entity embeddings from image for initialization all_obj_bert_embeddings, compute the centeriod with trasitional kmeans 
            all_obj_bert_embeddings_numpy = all_obj_bert_embeddings.cpu().numpy()
            print('[pretraining done]')
            kmeans_model = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(all_obj_bert_embeddings_numpy)
            print('[kmeans done]')
            print('[K-Means prediction for embeddings]')
            all_obj_symbolic_representation = kmeans_model.predict(all_obj_bert_embeddings_numpy)
            print(all_obj_symbolic_representation)
            print('[Labels]')
            print(self.all_entity_labels)
            if type(self.all_entity_labels[0]) == str:
                # Create a mapping: string → unique int
                str_to_int = {s: i for i, s in enumerate(sorted(set(self.all_entity_labels)))}
                # Map each string to its corresponding int
                int_list = [str_to_int[s] for s in self.all_entity_labels]
                print(int_list)  # e.g., [0, 1, 0, 2, 1]
                self.all_entity_labels_index = int_list
            else:
                self.all_entity_labels_index = self.all_entity_labels
            acc_initial = accuracy_score(kmeans_model.labels_, self.all_entity_labels_index)
            self.plot_cluster_label_confusion(y_true=self.all_entity_labels_index,y_pred=kmeans_model.labels_)
            print(f'[Accuracy of initial K-Means]')
            print(acc_initial)
            self.cluster.cluster_rep = torch.nn.Parameter(torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32, device=self.device), requires_grad=True)


        # Define parameter groups
        if self.data_format == 'image':
            parameter_groups = [
                {'params': self.logic_model.parameters(), 'lr': self.lr_rule_model},  # Lower learning rate for the first layer
                {'params': head_model.parameters(), 'lr': self.lr_predicate_model},  # Higher learning rate for the second layer
                {'params': self.cluster.parameters(), 'lr': self.lr_dkm}  # Higher learning rate for the second layer
            ]
        else:
            parameter_groups = [
                {'params': self.logic_model.parameters(), 'lr': self.lr_rule_model},  # Lower learning rate for the first layer
                {'params': head_model.parameters(), 'lr': self.lr_predicate_model}  # Higher learning rate for the second layer
            ]
        logic_optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)

        # load pre rules 
        try:
            with open(self.rule_path+'.pk', "rb") as f:
                rule_set = pickle.load(f)
                f.close()
        except FileNotFoundError:
            print(f"Rule file {self.rule_path+'.pk'} not found. Starting with an empty rule set.")
            rule_set = None

        # start training
        self.logic_model.train()
        head_model.train()
        infer_time = 0
        for single_epoch in range(self.epoch):
            total_loss_rule = 0
            total_loss = 0
            total_dkm_loss = 0
            total_body_loss = 0
            if self.data_format == 'image':
                self.cluster.train()
            #todo change the dataset for each epoch when learning from images, adjust the batch symbolic representations 
            if self.data_format == 'image':
                if self.substitution_method == 'random' or self.number_variable >= 3:
                    all_obj_bert_embeddings, all_obj_symbolic_representation, kmeans_loss = self.cluster(all_obj_bert_embeddings)
                    # update the knowledge base 
                    all_facts_bert_embeddings = self.update_all_fact_embedding_image(all_obj_bert_embeddings, selected_train_R)

                    tuple_data = BatchZ(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity, number_variable = self.number_variable)
                    substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True)

                else:
                    self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity_cluster(all_obj_symbolic_representation, train_data_E.shape[0])
                    tuple_data = BatchChain(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity,index_to_entity=self.index_to_entity, entity_to_entityindex=self.entity_entityindex, entity_to_index = self.entity_index)
                    substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True)
            
            for index in tqdm.tqdm(substation_data):
                if self.substitution_method == 'all':
                    index = torch.stack(index)
                    index = index.transpose(0,1)
                logic_optimizer.zero_grad()
                index = index.to(self.device).int()
                
                # when multiple dimension, turn them into binary siz 
                index = index.reshape(-1, index.shape[-1])
                
                # index = index.squeeze(0)
                # transform the index                 
                number_instance = index.shape[0]

                # # ! We only consider the X Y Z three variables in this stage
                # obtain the substitution for all atoms in body, get the all variable perturbation and return the substitutions based on X, Y, and Z variable
                all_row = torch.arange(0, index.shape[0], device=self.device).unsqueeze(1).expand(-1, self.variable_perturbations.shape[1])
                all_row = all_row.unsqueeze(1).expand(-1, self.variable_perturbations.shape[0],-1)
                variable_perturbations_variance = self.variable_perturbations.unsqueeze(0).expand(index.shape[0], -1, -1)
                perturbations = index[all_row, variable_perturbations_variance]
                
                # todo using perturbations to get the labels
                # combine the relation index to the perturbations 
                
                if self.data_format == 'kg':
                    pertubations_relations = perturbations.unsqueeze(2)
                    pertubations_relations_expand = pertubations_relations.expand(-1, -1, number_relations, -1)
                    unique_relation_index = self.unique_relation_index.expand(pertubations_relations_expand.shape[1], -1, -1)
                    unique_relation_index = unique_relation_index.unsqueeze(0).expand(pertubations_relations_expand.shape[0], -1, -1, -1)
                    combine_facts = torch.cat([pertubations_relations_expand, unique_relation_index], dim=3)
                    combine_facts = combine_facts.reshape(number_instance, -1, 3)
                    combine_facts_ready_check = combine_facts.reshape(-1, 3)
                    # check the occurrence
                    combine_facts_ready_check = combine_facts_ready_check.unsqueeze(1)
                    all_facts_var = self.all_fact_index.unsqueeze(0)
                    labels = (combine_facts_ready_check==all_facts_var).all(dim=2).any(dim=1)
                    labels = labels.reshape(number_instance, -1).float()
                    labels_body = labels
                
                elif self.data_format == 'image':
                    # first find labels for all perturbation 
                    # modify all_obj_bert_embeddings to centeriod embeddings
                    all_obj_bert_embeddings_centeriods, all_obj_symbolic_representation, kmeans_loss = self.cluster(all_obj_bert_embeddings)
                    # encoder = Encoder(768, 20)
                    # all_obj_bert_embeddings = encoder(all_obj_bert_embeddings)
                    
                    # update the knowledge base 
                    all_facts_bert_embeddings = self.update_all_fact_embedding_image(all_obj_bert_embeddings_centeriods, selected_train_R)
                    
                    
                    
                    #! Based on the all body predicates and entity pairs, generate the labels 
                    perturbations_bert_embeddings = all_obj_bert_embeddings_centeriods[perturbations]
                    # assign the clustering centroid to the entity and using the embedding of the centroid as the embedding of entities
                    perturbations_bert_embeddings = perturbations_bert_embeddings.unsqueeze(2)
                    expanded_per = perturbations_bert_embeddings.expand(-1, -1, number_relations, -1, -1)                
                    selected_r_bert_embeddings_var = selected_r_bert_embeddings.unsqueeze(0).expand(expanded_per.shape[0], -1, -1, -1, -1)
                    triples = torch.cat([expanded_per, selected_r_bert_embeddings_var], dim=3)
                    triples = triples.reshape(number_instance, -1, 3, self.bert_dim_length)
                    triples = triples.reshape(-1, 3, self.bert_dim_length)
                    first_shape = triples.shape[0]
                    triples = triples.reshape(first_shape,-1)
                    # check the occurrence 
                    triples = triples.unsqueeze(1)
                    all_facts_bert_embeddings_var = all_facts_bert_embeddings.unsqueeze(0)
                    labels_body = self.batched_row_existence(triples, all_facts_bert_embeddings_var)
                    # labels_body = (triples==all_facts_bert_embeddings_var).all(dim=2).any(dim=1)
                    labels_body = labels_body.reshape(number_instance, -1).float()
                
                
                # assert torch.allclose(labels, labels_body, rtol=1e-05, atol=1e-08), "Tensors are not equal"
                # get the head embeddings for all perturbations 
                # todo the number of false head is missing 
                # todo all body are all zero because the limited z 
                
                # get the head embedding of the entity 
                if self.data_format == 'image':
                    #! for image task, we need to use the cluster centeriod as the entity embedding 
                    selected_train_E_head_embeddings, clusters1, l1 = self.cluster(selected_train_E)
                    selected_train_L_head_embeddings, clusters2, l2 = self.cluster(selected_train_L)
                    selected_train_E_head_embeddings = head_model.generate_embedding(selected_train_E_head_embeddings)
                    selected_train_L_head_embeddings = head_model.generate_embedding(selected_train_L_head_embeddings)
                else:
                    selected_train_E_head_embeddings = head_model.generate_embedding(selected_train_E)
                    selected_train_L_head_embeddings = head_model.generate_embedding(selected_train_L)
                selected_r_head_embeddings = head_model.generate_embedding(original_shape_selected_r_bert_embeddings)
                all_objects_head_embedding = torch.cat([selected_train_E_head_embeddings, selected_train_L_head_embeddings], dim=0).to(self.device)
                selected_r_head_embeddings = selected_r_head_embeddings.unsqueeze(0)
                selected_r_head_embeddings  = selected_r_head_embeddings.expand(self.number_variable_terms, -1, -1)
                selected_r_head_embeddings = selected_r_head_embeddings.unsqueeze(2)
                
                perturbations_embeddings = all_objects_head_embedding[perturbations.int()]
                perturbations_embeddings = perturbations_embeddings.unsqueeze(2)
                expanded_per = perturbations_embeddings.expand(-1, -1, number_relations, -1, -1)
                all_r_embeddings_variance = selected_r_head_embeddings.unsqueeze(0).expand(expanded_per.shape[0], -1, -1, -1, -1)
                # ! Combine the relation to all entities 
                triples = torch.cat([expanded_per, all_r_embeddings_variance], dim=3)    
                triples = triples.reshape(number_instance, -1, 3, self.final_dim) # R_0(X,Y), R_1(X,Y), ...,R_n(X,Y) ,R_0(Y,X),.., R_n(Y,X), ... X (e1, e2, r) X embedding length 
                triples = triples.reshape(-1, 3, self.final_dim)
                
                # get the distance between the entity and relation
                e_hidden_states = triples[:,0,:]
                r_hidden_states = triples[:,2,:]
                l_hidden_states = triples[:,1,:]
                ground_truth_values = head_model.get_possible(e1 = e_hidden_states,  embed_r =  r_hidden_states, e2 = l_hidden_states)
                ground_truth_values = ground_truth_values.reshape(number_instance, -1)

                # ground_truth_values = ground_truth_values.reshape(-1)
                # labels_body = labels_body.reshape(-1)
                
                # get valid atom labels and ground truth values
                if self.open_neural_predicate == True:
                    ground_truth_values = ground_truth_values[:,self.valid_atom_index]
                    labels_body = labels_body[:,self.valid_atom_index]
                    body_loss = torch.nn.BCELoss()(ground_truth_values, labels_body) # can use Cross Entropy Loss
                else:
                    # todo the valid atom index can be updated based on the index from clustering 
                    labels_body = labels_body[:,self.valid_atom_index]
                    ground_truth_values = labels_body
                    body_loss = 0
                head_predicate_labels_predicated = ground_truth_values[:,self.target_atom_index].unsqueeze(1)
                body_predicate_labels_predicated = torch.cat([ground_truth_values[:,:self.target_atom_index], ground_truth_values[:,self.target_atom_index+1:]],dim=1)

                # using logic model and body atoms to predict the head predicate
                predicated = self.logic_model(body_predicate_labels_predicated)
                loss_rule = torch.nn.MSELoss()(predicated, head_predicate_labels_predicated)

                #loss for dkm 
                if self.data_format == 'image' and self.open_neural_predicate == True:
                    dkm_loss = l1 + l2 + kmeans_loss
                elif self.data_format == 'image' and self.open_neural_predicate == False:
                    dkm_loss = kmeans_loss
                else: 
                    dkm_loss = 0
                
                # all loss from logic rule loss, body predicate, and [deep k-means]
                if self.open_neural_predicate == True:
                    loss = 0.5*loss_rule + 0.5*body_loss + 0.5 * dkm_loss
                else:
                    # loss = 0.5*loss_rule + 0.5 * dkm_loss
                    loss = loss_rule + self.lambda_dkm * dkm_loss
                

                loss.backward(retain_graph=True)
                # todo define the loss from neural predicate here 
                logic_optimizer.step()
                total_loss_rule = total_loss_rule + loss_rule.item()
                total_loss += loss.item()
                if self.open_neural_predicate == True:
                    total_loss += body_loss.item()
                else:
                    total_loss += 0
                if self.data_format == 'image':
                    total_dkm_loss += dkm_loss.item()
                else:
                    total_dkm_loss += 0

            # if the input is the image, build the knowledge graphs based on the clustering centroid as the entities and the existing relations as the predicates
            dkm_acc = 0
            if self.data_format == 'image':
                dkm_acc = self.build_kg_based_on_cluster(selected_train_E, selected_train_L, all_relations_in_order)
                # build on test data
                infer_time = self.infer(infer_time)
                

            total_loss_rule = total_loss_rule / len(substation_data)
            total_loss = total_loss / len(substation_data)
            total_dkm_loss = total_dkm_loss / len(substation_data)
            total_body_loss = total_body_loss / len(substation_data)
            print(f"Epoch: {single_epoch}, Loss: {total_loss}, Body Loss: {total_body_loss}, Rule Loss: {total_loss_rule}, DKM Loss: {total_dkm_loss}, DKM acc, {dkm_acc}")
            self.writer.add_scalar("Loss/train", total_loss, single_epoch)
            self.writer.add_scalar("Loss/train_Logic", total_loss_rule, single_epoch)
            self.writer.add_scalar("Loss/body_loss", total_body_loss, single_epoch)
            self.writer.add_scalar("Loss/dkm_loss", total_dkm_loss, single_epoch)
            self.writer.add_scalar("ACC/dkm_acc", dkm_acc, single_epoch)
            self.writer.flush()
            
            # compute the acc on the knowledge graphs 
            train_acc, rule_set = self.check_acc(rule_set)
            # save rule set 
            with open(self.rule_path+'.pk', "wb") as f:
                pickle.dump(rule_set, f)
                f.close()
            train_precision = train_acc['precision']
            train_recall = train_acc['recall']
            if train_recall >= self.stop_recall:
                break
        self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
        print("[Training Finished]")
        print("[Rules]")
        print(rule_set)
        print(f"[Precision]")
        print(train_precision)
        print(f"[Recall]")
        print(train_recall)
        return 0 
    
    def partial_PI(self, inner_batch = 2):
        '''
        Read all entities and process and propositionalalization and rule learning task together.
        Train distance and logic model together.
        BERT embedding is fix but head embeddings is trainable.
        '''
        self.number_variable = self.n_clusters
        # consider the binary predicate and unary predicate at the same time 
        if self.body_arity == '12':
            self.variable_perturbations_list = list(itertools.combinations(range(self.number_variable), 2)) + [(x,x) for x in range(self.number_variable)]
        elif self.body_arity == '1':
            self.variable_perturbations_list = [(x,x) for x in range(self.number_variable)]
        elif self.body_arity == '2':
            self.variable_perturbations_list = list(itertools.combinations(range(self.number_variable), 2))
        else:
            raise ValueError("The body arity should be 1, 2, or 12")
        self.variable_perturbations = torch.tensor(self.variable_perturbations_list, device=self.device)
        
        # for predicate atoms. 
        self.variable_perturbations_list_with_predicate = list(itertools.permutations(range(self.number_variable), 2)) + [(x,x) for x in range(self.number_variable)]
        self.variable_perturbations_with_predicate = torch.tensor(self.variable_perturbations_list_with_predicate, device=self.device)
        
        self.all_unground_atoms = []
        self.all_relations = [''] # placeholder for the invented predicate. For each tuple of variable, only one invented predicate will be created

        # TODO get the partial relations KB
        train_data = load_dataset('csv',  data_files=f"{self.current_dir}/cache/{self.task_name}/train_embeddings.csv")['train']
        self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity(train_data)
        all_relations_in_order = [i for i in train_data['textR']]
        all_relation_occurrence_index = {}
        unique_relations  = 0
        for i in all_relations_in_order:
            if i not in all_relation_occurrence_index:
                all_relation_occurrence_index[i] = unique_relations
                unique_relations += 1

        
        # self.all_relation_index = torch.tensor([i for i in range(len(all_relations_in_order))]).to(self.device)


        # self.all_relation_mapping = []
        # for i in train_data['textR']:
        #     self.all_relation_mapping.append(all_relation_occurrence_index[i])
        # self.all_relation_mapping = torch.tensor(self.all_relation_mapping).to(self.device)
        
        # self.all_relation_index = self.all_relation_mapping[self.all_relation_index].unsqueeze(0)

        # self.all_first_entity = [self.return_representative_index(i) for i in range(len(train_data['text1']))]
        # self.all_first_entity = torch.tensor(self.all_first_entity).unsqueeze(0).to(self.device)
        # self.all_second_entity = [self.return_representative_index(i) for i in range(len(train_data['text1']), 2*len(train_data['text1']) )]
        # self.all_second_entity = torch.tensor(self.all_second_entity).unsqueeze(0).to(self.device)


        # self.all_fact_index = torch.concat([self.all_first_entity, self.all_second_entity, self.all_relation_index], dim=0).T
        
        # self.unique_relation_index = torch.unique(self.all_relation_index, sorted=False).unsqueeze(0)
        # self.unique_relation_index.expand(self.number_variable_terms, -1)
        # self.unique_relation_index = self.unique_relation_index.unsqueeze(2)
        
        
        self.partial_relations = OrderedSet(train_data['textR'])
        self.all_relations.extend(self.partial_relations)

        #!  get target predicate in the KB dataset for batch find indexing
        target_atoms_index = []
        ini_index = 0
        for item in train_data['textR']:
            if item == self.target_relation:
                target_atoms_index.append(ini_index)
            ini_index = ini_index + 1
        



        # TODO get valid unground atoms 
        # get predicate arity 
        predicate_arity = self.check_arity_predicate(train_data)
        print('Predicate arity:', predicate_arity)
        

        # get the arity of the target predicate 
        self.target_predicate_arity = predicate_arity[self.target_relation]
        
        # get all atom arrangement 
        self.all_unground_atoms = []
        for arrage in self.variable_perturbations_list:
            self.all_unground_atoms.append(f"[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")  
        for arrage in self.variable_perturbations_list_with_predicate:
            for item in self.partial_relations:
                self.all_unground_atoms.append(f"{item}[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")
        target_index = self.all_unground_atoms.index(f'{self.target_relation}[{self.target_variable_arrange}]')
        
        # get the valid list and valid unground atoms 
        invalid_atom_index = []
        valid_atom_index = []
        for ele in range(len(self.all_unground_atoms)):
            item = self.all_unground_atoms[ele]
            predicate = item.split('[')[0]
            first_variable = item.split('[')[1].split('@')[0]
            second_variable = item.split('[')[1].split('@')[1].split(']')[0]
            if first_variable == second_variable:
                atom_variable = 1
            else:
                atom_variable = 2
            if predicate == '':
                valid_atom_index.append(ele)
                continue
            if (predicate_arity[predicate] == 2 and atom_variable == 1) or (predicate_arity[predicate] == 1 and atom_variable == 2):
                invalid_atom_index.append(ele)
            else:
                valid_atom_index.append(ele)
        self.valid_atom_index = valid_atom_index
        self.invalid_atom_index = invalid_atom_index
        self.valid_unground_atoms = []
        for ele in valid_atom_index:
            self.valid_unground_atoms.append(self.all_unground_atoms[ele])
        print('Valid unground atoms:', self.valid_unground_atoms)
        
        # ! find and update the target atom index in the valid unground atoms
        self.target_atom_index = self.valid_unground_atoms.index(f'{self.target_relation}[{self.target_variable_arrange}]')


        
        # load the positive and negative data and an random index 
        # ! insert the testing dataset here 
        train_data_pos = torch.load(f"{self.current_dir}/cache/{self.task_name}/positive_images_data_train.pt")
        # train_data_neg = torch.load(f"{self.current_dir}/cache/{self.task_name}/negative_images_data_train.pt")
        # test_data_pos = torch.load(f"{self.current_dir}/cache/{self.task_name}/positive_images_data_test.pt")
        # test_data_neg = torch.load(f"{self.current_dir}/cache/{self.task_name}/negative_images_data_test.pt")
        # all_obj_bert_embeddings = torch.concat([train_data_pos,train_data_neg], dim=0).to(self.device) # get all true index of object
        train_data_pos = torch.tensor(train_data_pos).to(self.device) # get all true index of fact

        # TODO  load facts for KB 
        train_data_E = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_E_10.pt").to(self.device)
        train_data_R = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_R_10.pt").to(self.device)
        train_data_L = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_L_10.pt").to(self.device)
        all_fact_bert_embeddings = torch.cat([train_data_E, train_data_L, train_data_R], dim=1) # get all true index of fact
        all_obj_bert_embeddings = torch.cat([train_data_E, train_data_L], dim=0).to(self.device) # get all true index of object

        #todo Unique the relation bert embeddings and keep the first occurrence as the index 
        # Step 1: Track unique rows and map them to indices
        seen = {}
        unique_rows = []
        inverse_indices = []

        for row in train_data_R:
            row_tuple = tuple(row.tolist())
            if row_tuple not in seen:
                seen[row_tuple] = len(seen)  # assign new index
                unique_rows.append(row)
            inverse_indices.append(seen[row_tuple])

        # Step 2: Convert to tensors
        unique_tensor = torch.stack(unique_rows)               # (num_unique, D)
        inverse_tensor = torch.tensor(inverse_indices)         # (N,)

        print("Original tensor:\n", train_data_R)
        print("Unique rows (first occurrence order):\n", unique_tensor)
        print("Inverse indices:\n", inverse_tensor)
        # all_relation_bert_embedding [44,768]
        selected_r_bert_embeddings = unique_tensor.unsqueeze(0)
        original_shape_selected_r_bert_embeddings = unique_tensor
        selected_r_bert_embeddings  = selected_r_bert_embeddings.expand(len(self.variable_perturbations_list_with_predicate), -1, -1)
        selected_r_bert_embeddings = selected_r_bert_embeddings.unsqueeze(2)

        # choose all substitution method for multiple variable in the settings 
        # tuple_data = AllSubstitution(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity,index_to_entity=self.index_to_entity, entity_to_index=self.entity_index)
        # substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=self.random_negative, shuffle=True)
            
        
        # load the index to image file 
        with open(f"{self.current_dir}/cache/{self.task_name}/index_to_image_train.pkl", 'rb') as f:
            self.index_to_image = pickle.load(f)
            f.close()
        

        # 1. Load neural logic rule learning model 
        self.logic_model = DeepRuleLayer(in_size=len(self.valid_unground_atoms)-1, rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=self.rule_path, t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=self.target_arity,time_stamp='')
        self.logic_model.to(self.device)
        
        
        # 2. Define the differentiable k-means module for image task
        self.cluster = DkmCompGraph(ae_specs=self.bert_dim_length, n_clusters=self.n_clusters, val_lambda=1, input_size=None, alpha=self.alpha,device=self.device, mode='without_vae')
        # with all entity embeddings from image for initialization all_obj_bert_embeddings, compute the centeriod with trasitional kmeans 
        # reshape embedding into -1 768
        all_obj_bert_embeddings_numpy = all_obj_bert_embeddings.view(-1, self.bert_dim_length).cpu().numpy()  # Reshape to (-1, 768)
        all_image_embeddings_numpy = train_data_pos.reshape(-1, self.bert_dim_length).cpu().numpy()  # Reshape to (-1, 768)
        print('[pretraining done]')
        kmeans_model = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(all_obj_bert_embeddings_numpy)
        print('[kmeans done]')
        print('[K-Means prediction for embeddings]')
        all_obj_symbolic_representation = kmeans_model.predict(all_image_embeddings_numpy)
        print(all_obj_symbolic_representation)
        # self.plot_cluster(all_obj_bert_embeddings_numpy, cluster_labels = all_obj_symbolic_representation, addition_inputs = 'init_')

            
        # print(f'[Accuracy of initial K-Means]')
        # print(acc_initial)
        self.cluster.cluster_rep = torch.nn.Parameter(torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32, device=self.device), requires_grad=True)

        # Define parameter groups
        parameter_groups = [
            {'params': self.logic_model.parameters(), 'lr': self.lr_rule_model},  # Lower learning rate for the first layer
            {'params': self.cluster.parameters(), 'lr': self.lr_dkm}  # Higher learning rate for the second layer
        ]
        logic_optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)

        #todo change the dataset for each epoch when learning from images, adjust the batch symbolic    representations 
        # self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity_cluster(all_obj_symbolic_representation, train_data_E.shape[0])
        tuple_data = Batch_PartialPI(train_data_pos, number_variable = self.number_variable, inner_batch = inner_batch, entity_path=train_data_E, relation_path = train_data_R, label_path = train_data_L,random_negative = self.random_negative, target_predicate_arity=self.target_predicate_arity, target_atoms_index=target_atoms_index)
        substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True) 
        # start training
        self.logic_model.train()
        self.cluster.train()
        rule_set = None
        infer_time = 0
        self.inner_batch = inner_batch
        for single_epoch in range(self.epoch):
            total_loss_rule = 0
            total_loss = 0
            total_dkm_loss = 0
            total_body_loss = 0
            
            for index, fact_index in tqdm.tqdm(substation_data):
                
                # todo get the image embeddings 
                logic_optimizer.zero_grad()
                index = index.to(self.device)
                fact_index  = fact_index.to(self.device)
                # index = index.squeeze(0)
                # transform the index             
                first_shape = index.shape[0]
                all_data = index.reshape(-1, self.bert_dim_length)
                
                # get the symbolic representation for the embeddings 
                all_obj_bert_embeddings_centeriods, all_obj_symbolic_representation_diff, kmeans_loss = self.cluster(all_data)
                # all_data_numpy = all_data.cpu().numpy()
                # all_obj_symbolic_representation = torch.tensor(kmeans_model.predict(all_data_numpy), device=self.device)
                # print(all_obj_symbolic_representation)
                # print(all_obj_symbolic_representation_diff)
                all_obj_symbolic_representation = all_obj_symbolic_representation_diff.reshape(first_shape, self.inner_batch,-1)
                # sort each element in all_obj_symbolic_representation in the second dimension
                # get all combinations of two elements in each row
                n = all_obj_symbolic_representation.size(-1)
                # Get upper-triangle indices (combinations of 2 elements without repetition)
                idx = torch.triu_indices(n, n, offset=0)
                # Get the two elements from each row using the indices
                comb1 = all_obj_symbolic_representation[:, :, idx[0]]  # Shape: (batch_size, num_combinations)
                comb2 = all_obj_symbolic_representation[:, :, idx[1]]  # Same
                # Stack along a new dimension to form pairs
                binary_facts = torch.stack((comb1, comb2), dim=3)  # Shape: (batch_size, num_combinations, 2)
                # sort in the third dimension
                all_facts = torch.sort(binary_facts, dim=-1)[0]
                all_facts_unique = all_facts.unique(dim=2)  # Remove duplicate pairs across the batch
                
                # check the template is in the facts or not 
                # Step 1: Reshape A to (1, n, d) and B to (m, n, d)
                # We'll compare each row in A to all rows in B across axis
                A = self.variable_perturbations  # shape (1, n, d)
                B = all_facts_unique  # shape (m, n, d)

                A_exp = A.unsqueeze(0).unsqueeze(2)      # [1, 6, 1, 2]
                B_exp = B.unsqueeze(2)                   # [3, 1, 3, 2]

                # Compare each A[i] with all elements in each B[b]
                matches = (A_exp == B_exp).all(dim=-1)   # [1, 6, 3]

                # Check if A[i] exists in any row in B[b]
                exists = matches.any(dim=-1).unsqueeze(-1)  # [1, 6, 1]

                # print(exists)
                
                # using logic model and body atoms to predict the head predicate
                body_predicate_labels_predicated = exists.reshape(first_shape, self.inner_batch, -1).float()
                labels_body_placeholder = body_predicate_labels_predicated.reshape(first_shape*self.inner_batch, -1)  # Reshape to match the label shape



                # todo get the fact with predicate
                fact_index = fact_index.squeeze(0).long()
                number_instance = fact_index.shape[0]

                all_row = torch.arange(0, fact_index.shape[0], device=self.device).unsqueeze(1).expand(-1, self.variable_perturbations_with_predicate.shape[1])

                all_row = all_row.unsqueeze(1).expand(-1, self.variable_perturbations_with_predicate.shape[0],-1)

                variable_perturbations_variance = self.variable_perturbations_with_predicate.unsqueeze(0).expand(fact_index.shape[0], -1, -1)
                
                perturbations = fact_index[all_row, variable_perturbations_variance]

                # TODO for image check 
                # first find labels for all perturbation 
                # modify all_obj_bert_embeddings to centeriod embeddings
                all_obj_bert_embeddings_centeriods, all_obj_symbolic_representation, kmeans_loss = self.cluster(all_obj_bert_embeddings)
                # encoder = Encoder(768, 20)
                # all_obj_bert_embeddings = encoder(all_obj_bert_embeddings)
                
                # update the knowledge base 
                all_facts_bert_embeddings = self.update_all_fact_embedding_image(all_obj_bert_embeddings_centeriods, train_data_R)
                
                
                
                #! Based on the all body predicates and entity pairs, generate the labels 
                perturbations_bert_embeddings = all_obj_bert_embeddings_centeriods[perturbations]
                # assign the clustering centroid to the entity and using the embedding of the centroid as the embedding of entities
                perturbations_bert_embeddings = perturbations_bert_embeddings.unsqueeze(2)
                expanded_per = perturbations_bert_embeddings.expand(-1, -1, len(self.partial_relations), -1, -1)                
                selected_r_bert_embeddings_var = selected_r_bert_embeddings.unsqueeze(0).expand(expanded_per.shape[0], -1, -1, -1, -1)
                triples = torch.cat([expanded_per, selected_r_bert_embeddings_var], dim=3)
                triples = triples.reshape(number_instance, -1, 3, self.bert_dim_length)
                triples = triples.reshape(-1, 3, self.bert_dim_length)
                first_shape = triples.shape[0]
                triples = triples.reshape(first_shape,-1)
                # check the occurrence 
                triples = triples.unsqueeze(1)
                all_facts_bert_embeddings_var = all_facts_bert_embeddings.unsqueeze(0)
                labels_body = self.batched_row_existence(triples, all_facts_bert_embeddings_var)
                # labels_body = (triples==all_facts_bert_embeddings_var).all(dim=2).any(dim=1)
                labels_body_predicate = labels_body.reshape(number_instance, -1).float()




                # pertubations_relations = perturbations.unsqueeze(2)
                # pertubations_relations_expand = pertubations_relations.expand(-1, -1, len(self.partial_relations), -1)
                # unique_relation_index = self.unique_relation_index.expand(pertubations_relations_expand.shape[1], -1, -1)
                # unique_relation_index = unique_relation_index.unsqueeze(0).expand(pertubations_relations_expand.shape[0], -1, -1, -1)
                # combine_facts = torch.cat([pertubations_relations_expand, unique_relation_index], dim=3)
                # combine_facts = combine_facts.reshape(number_instance, -1, 3)
                # combine_facts_ready_check = combine_facts.reshape(-1, 3)
                # # check the occurrence
                # combine_facts_ready_check = combine_facts_ready_check.unsqueeze(1)
                # all_facts_var = self.all_fact_index.unsqueeze(0)
                # labels = (combine_facts_ready_check==all_facts_var).all(dim=2).any(dim=1)
                # labels = labels.reshape(number_instance, -1).float()
                # labels_body_predicate = labels

                # number of placehoder and predicate 
                number_placeholder = labels_body_placeholder.shape[0]
                number_predicate = labels_body_predicate.shape[0]
                # concat  label from placeholder and predicate together 
                labels_body_placeholder_expanded = labels_body_placeholder.unsqueeze(1).repeat(1, number_predicate, 1)  # shape: (2, 3, 10)

                # Repeat B for each row in A
                number_predicate_expanded = labels_body_predicate.unsqueeze(0).repeat(number_placeholder, 1, 1)  # shape: (2, 3, 5)

                # Concatenate along last dimension
                merged = torch.cat([labels_body_placeholder_expanded, number_predicate_expanded], dim=-1)  # shape: (2, 3, 15)

                # Reshape to (6, 15)
                labels_body = merged.view(-1, merged.shape[-1]) 


                # concat the body predicate labels and the ground truth values 
                ground_truth_values  = labels_body[:, self.valid_atom_index]  # get the valid atom index for the body predicate
                head_predicate_labels = ground_truth_values[:,self.target_atom_index].unsqueeze(1)
                body_predicate_labels_predicated = torch.cat([ground_truth_values[:,:self.target_atom_index], ground_truth_values[:,self.target_atom_index+1:]],dim=1)




                # Todo get the body atom values 
                predicated = self.logic_model(body_predicate_labels_predicated)
                label = head_predicate_labels
                # label = label.reshape(first_shape*self.inner_batch, -1)  # Reshape to match the predicated shape
                loss_rule = torch.nn.MSELoss()(predicated, label)
                loss_rule = loss_rule.float()
                
                # all loss from logic rule loss, body predicate, and [deep k-means]
                loss = 0.5*loss_rule + 0.5 * kmeans_loss
                
                loss.backward(retain_graph=True)
                logic_optimizer.step()
                total_loss_rule += loss_rule.item()
                total_loss += loss.item()
                total_dkm_loss += kmeans_loss.item()


            # if the input is the image, build the knowledge graphs based on the clustering centroid as the entities and the existing relations as the predicates
            dkm_acc = 0

            # build on test data
            # infer_time = self.infer(infer_time)
            self.plot_cluster_ppi(all_obj_bert_embeddings)
            total_loss_rule = total_loss_rule / len(substation_data)
            total_loss = total_loss / len(substation_data)
            total_dkm_loss = total_dkm_loss / len(substation_data)
            print(f"Epoch: {single_epoch}, Loss: {total_loss}, Body Loss: {total_body_loss}, Rule Loss: {total_loss_rule}, DKM Loss: {total_dkm_loss}, DKM acc, {dkm_acc}")
            self.writer.add_scalar("Loss/train", total_loss, single_epoch)
            self.writer.add_scalar("Loss/train_Logic", total_loss_rule, single_epoch)
            self.writer.add_scalar("Loss/body_loss", total_body_loss, single_epoch)
            self.writer.add_scalar("Loss/dkm_loss", total_dkm_loss, single_epoch)
            self.writer.add_scalar("ACC/dkm_acc", dkm_acc, single_epoch)
            self.writer.flush()
            
            # compute the acc on the knowledge graphs 
            # todo consider how to add it 
            self.build_kg_ppi(train_data, train_data_E, train_data_L)
            acc, rule_set = self.check_acc(rule_set)
            # save rule set 
            with open(self.rule_path+'.pk', "wb") as f:
                pickle.dump(rule_set, f)
                f.close()
            # todo consider how to add it 
            precision = acc['precision']
            recall = acc['recall']
            print('single epoch recall', recall)
            if recall >= self.stop_recall:
                break
        self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
        print("[Training Finished]")
        print("[Rules]")
        print(rule_set)
        print(f"Test Precision")
        print(precision)
        print(f"Test Recall")
        print(recall)
        return 0 
    
    def predicate_invention(self, inner_batch = 2):
        '''
        Read all entities and process and propositionalalization and rule learning task together.
        Train distance and logic model together.
        BERT embedding is fix but head embeddings is trainable.
        '''
        self.number_variable = self.n_clusters
        # consider the binary predicate and unary predicate at the same time 
        if self.body_arity == '12':
            self.variable_perturbations_list = list(itertools.combinations(range(self.number_variable), 2)) + [(x,x) for x in range(self.number_variable)]
        elif self.body_arity == '1':
            self.variable_perturbations_list = [(x,x) for x in range(self.number_variable)]
        elif self.body_arity == '2':
            self.variable_perturbations_list = list(itertools.combinations(range(self.number_variable), 2))
        else:
            raise ValueError("The body arity should be 1, 2, or 12")
        self.variable_perturbations = torch.tensor(self.variable_perturbations_list, device=self.device)
        self.all_unground_atoms = []
        self.all_relations = [''] # placeholder for the invented predicate. For each tuple of variable, only one invented predicate will be created
        for arrage in self.variable_perturbations_list:
            for item in self.all_relations:
                self.all_unground_atoms.append(f"{item}[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")
        self.valid_unground_atoms = self.all_unground_atoms
        self.inner_batch = inner_batch
        
        # load the positive and negative data and an random index 
        train_data_pos = torch.load(f"{self.current_dir}/cache/{self.task_name}/positive_images_data_train.pt")
        train_data_neg = torch.load(f"{self.current_dir}/cache/{self.task_name}/negative_images_data_train.pt")
        test_data_pos = torch.load(f"{self.current_dir}/cache/{self.task_name}/positive_images_data_test.pt")
        test_data_neg = torch.load(f"{self.current_dir}/cache/{self.task_name}/negative_images_data_test.pt")
        all_obj_bert_embeddings = torch.concat([train_data_pos, train_data_neg], dim=0).to(self.device) # get all true index of object
        
        # load the index to image file 
        with open(f"{self.current_dir}/cache/{self.task_name}/index_to_image_train.pkl", 'rb') as f:
            self.index_to_image = pickle.load(f)
            f.close()
        

        # 1. Load neural logic rule learning model 
        self.logic_model = DeepRuleLayer(in_size=len(self.all_unground_atoms), rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=self.rule_path, t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=self.target_arity,time_stamp='')
        self.logic_model.to(self.device)
        
        
        # 2. Define the differentiable k-means module for image task
        self.cluster = DkmCompGraph(ae_specs=self.bert_dim_length, n_clusters=self.n_clusters, val_lambda=1, input_size=None, alpha=self.alpha,device=self.device, mode='without_vae')
        # with all entity embeddings from image for initialization all_obj_bert_embeddings, compute the centeriod with trasitional kmeans 
        # reshape embedding into -1 768
        all_obj_bert_embeddings_numpy = all_obj_bert_embeddings.view(-1, self.bert_dim_length).cpu().numpy()  # Reshape to (-1, 768)
        print('[pretraining done]')
        kmeans_model = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(all_obj_bert_embeddings_numpy)
        print('[kmeans done]')
        print('[K-Means prediction for embeddings]')
        all_obj_symbolic_representation = kmeans_model.predict(all_obj_bert_embeddings_numpy)
        print(all_obj_symbolic_representation)
        self.plot_cluster(all_obj_bert_embeddings_numpy, self.index_to_image, cluster_labels = all_obj_symbolic_representation, addition_inputs = 'init_')

            
        # print(f'[Accuracy of initial K-Means]')
        # print(acc_initial)
        self.cluster.cluster_rep = torch.nn.Parameter(torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32, device=self.device), requires_grad=True)

        # Define parameter groups
        parameter_groups = [
            {'params': self.logic_model.parameters(), 'lr': self.lr_rule_model},  # Lower learning rate for the first layer
            {'params': self.cluster.parameters(), 'lr': self.lr_dkm}  # Higher learning rate for the second layer
        ]
        logic_optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)


        # start training
        self.logic_model.train()
        rule_set = None
        infer_time = 0
        for single_epoch in range(self.epoch):
            total_loss_rule = 0
            total_loss = 0
            total_dkm_loss = 0
            total_body_loss = 0

            self.cluster.train()
            #todo change the dataset for each epoch when learning from images, adjust the batch symbolic representations 
            # self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity_cluster(all_obj_symbolic_representation, train_data_E.shape[0])
            tuple_data = Batch_PI(train_data_pos, train_data_neg, inner_batch = self.inner_batch)
            substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=self.batch_size, shuffle=True)
            
            for index, label in tqdm.tqdm(substation_data):
                logic_optimizer.zero_grad()
                index = index.to(self.device)
                label  = label.to(self.device).float()
                # index = index.squeeze(0)
                # transform the index        
                entities_number = index.shape[1]
                #! take special attention why 3 is chosen 
                index_inedx = torch.randint(0, entities_number, (self.inner_batch, 3)).to(self.device) # current the v
                # select on each binary matrix 
                x = index 
                idx = index_inedx
                # Expand idx for batch dimension
                # [28, 3] -> [1024, 28, 3]
                idx_exp = idx.unsqueeze(0).expand(x.size(0), -1, -1)

                # Build batch indices [1024, 28, 3]
                batch_idx = torch.arange(x.size(0)).view(-1, 1, 1).expand_as(idx_exp)

                # Use advanced indexing
                out = x[batch_idx, idx_exp]   # [1024, 28, 3, 768]
                
                # index = index[index_inedx] 
                index = out 
                first_shape = index.shape[0]
                all_data = index.reshape(-1, self.bert_dim_length)
                
                # get the symbolic representation for the embeddings 
                all_obj_bert_embeddings_centeriods, all_obj_symbolic_representation_diff, kmeans_loss = self.cluster(all_data)
                # all_data_numpy = all_data.cpu().numpy()
                # all_obj_symbolic_representation = torch.tensor(kmeans_model.predict(all_data_numpy), device=self.device)
                # print(all_obj_symbolic_representation)
                # print(all_obj_symbolic_representation_diff)
                all_obj_symbolic_representation = all_obj_symbolic_representation_diff.reshape(first_shape, self.inner_batch,-1)
                # sort each element in all_obj_symbolic_representation in the second dimension
                # get all combinations of two elements in each row
                n = all_obj_symbolic_representation.size(-1)
                # Get upper-triangle indices (combinations of 2 elements without repetition)
                idx = torch.triu_indices(n, n, offset=0)
                # Get the two elements from each row using the indices
                comb1 = all_obj_symbolic_representation[:, :, idx[0]]  # Shape: (batch_size, num_combinations)
                comb2 = all_obj_symbolic_representation[:, :, idx[1]]  # Same
                # Stack along a new dimension to form pairs
                binary_facts = torch.stack((comb1, comb2), dim=3)  # Shape: (batch_size, num_combinations, 2)
                # sort in the third dimension
                all_facts = torch.sort(binary_facts, dim=-1)[0]
                all_facts_unique = all_facts.unique(dim=2)  # Remove duplicate pairs across the batch
                
                # check the template is in the facts or not 
                # Step 1: Reshape A to (1, n, d) and B to (m, n, d)
                # We'll compare each row in A to all rows in B across axis
                A = self.variable_perturbations  # shape (1, n, d)
                B = all_facts_unique  # shape (m, n, d)

                A_exp = A.unsqueeze(0).unsqueeze(2)      # [1, 6, 1, 2]
                B_exp = B.unsqueeze(2)                   # [3, 1, 3, 2]

                # Compare each A[i] with all elements in each B[b]
                matches = (A_exp == B_exp).all(dim=-1)   # [1, 6, 3]

                # Check if A[i] exists in any row in B[b]
                exists = matches.any(dim=-1).unsqueeze(-1)  # [1, 6, 1]

                # print(exists)
                
                # using logic model and body atoms to predict the head predicate
                body_predicate_labels_predicated = exists.reshape(first_shape, self.inner_batch, -1).float()
                body_predicate_labels_predicated = body_predicate_labels_predicated.reshape(first_shape*self.inner_batch, -1)  # Reshape to match the label shape
                predicated = self.logic_model(body_predicate_labels_predicated)
                label = label.unsqueeze(1).repeat(1, self.inner_batch)  # Repeat label to match the batch size
                label = label.reshape(first_shape*self.inner_batch, -1)  # Reshape to match the predicated shape
                loss_rule = torch.nn.MSELoss()(predicated, label)
                loss_rule = loss_rule.float()
                
                # all loss from logic rule loss, body predicate, and [deep k-means]
                loss = 0.5*loss_rule + 0.5 * kmeans_loss
                
                loss.backward(retain_graph=True)
                logic_optimizer.step()
                total_loss_rule += loss_rule.item()
                total_loss += loss.item()
                total_dkm_loss += kmeans_loss.item()


            # if the input is the image, build the knowledge graphs based on the clustering centroid as the entities and the existing relations as the predicates
            dkm_acc = 0

            # build on test data
            # infer_time = self.infer(infer_time)
            self.plot_cluster(all_obj_bert_embeddings, self.index_to_image)
            total_loss_rule = total_loss_rule / len(substation_data)
            total_loss = total_loss / len(substation_data)
            total_dkm_loss = total_dkm_loss / len(substation_data)
            print(f"Epoch: {single_epoch}, Loss: {total_loss}, Body Loss: {total_body_loss}, Rule Loss: {total_loss_rule}, DKM Loss: {total_dkm_loss}, DKM acc, {dkm_acc}")
            self.writer.add_scalar("Loss/train", total_loss, single_epoch)
            self.writer.add_scalar("Loss/train_Logic", total_loss_rule, single_epoch)
            self.writer.add_scalar("Loss/body_loss", total_body_loss, single_epoch)
            self.writer.add_scalar("Loss/dkm_loss", total_dkm_loss, single_epoch)
            self.writer.add_scalar("ACC/dkm_acc", dkm_acc, single_epoch)
            self.writer.flush()
            
            # compute the acc on the knowledge graphs 
            # todo consider how to add it 
            test_data = self.build_kb_based_on_cluster_predicate_invention(test_negative=test_data_neg, test_positive=test_data_pos, cluster_model=kmeans_model, kmtype='non_differentiable')
            acc, rule_set = self.check_acc_PI(test_data, rule_set)
            # save rule set 
            with open(self.rule_path+'.pk', "wb") as f:
                pickle.dump(rule_set, f)
                f.close()
            # todo consider how to add it 
            precision = acc['precision']
            recall = acc['recall']
            if recall >= self.stop_recall:
                break
        self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
        print("[Training Finished]")
        print("[Rules]")
        print(rule_set)
        print(f"Test Precision")
        print(precision)
        print(f"Test Recall")
        print(recall)
        print("Test Acc")
        print(acc['acc'])
        return 0 
    

    def rule_body_train_autoencoder(self):
        '''
        Read all entities and process and propositionalalization and rule learning task together.
        Train distance and logic model together.
        BERT embedding is fix but head embeddings is trainable.
        '''
        # read all entities 
        metadata_file =  'gammaILP/ILPdata/' + self.target_relation + '/train.pl'
        # initial image transform and text transform 
        img_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

        dataste_relation_image = TextTwoMNISTDataset(metadata_file, img_transform=img_transform, device = self.device)
        dataloader = torch.utils.data.DataLoader(dataste_relation_image, batch_size=64, shuffle=False)
        # pre_train image autoencode 
        self.ae = MNISTAutoencoder(latent_dim=768).to(self.device) 
        ae_optimizer = torch.optim.Adam(self.ae.parameters(), lr=1e-3)
        # autoencoder training time 
        pre_train_ae_num_epochs = 50
        for epoch in range(pre_train_ae_num_epochs):
            for item in dataloader:
                image_1 = item['img1'].to(self.device)
                image_2 = item['img2'].to(self.device)
                reconstructed_1, embeddings_1 = self.ae(image_1)
                reconstructed_2, embeddings_2 = self.ae(image_2)
                # compute loss 
                loss = nn.MSELoss()(reconstructed_1, image_1) + nn.MSELoss()(reconstructed_2, image_2)
                ae_optimizer.zero_grad()
                loss.backward()
                ae_optimizer.step()
            print(f"Epoch [{epoch+1}/{pre_train_ae_num_epochs}], Loss: {loss.item():.4f}")
        
        # Get embeddings from the encoder part of the autoencoder
        # all images into their corresponding embeddings
        all_embeddings_1 = []
        all_embeddings_2 = []
        all_embeddings_relation = []
        # self.index_to_image = []
        with torch.no_grad():
            for re in dataloader:
                image_1 = re['img1'].to(self.device)
                image_2 = re['img2'].to(self.device)
                relation_emb = re['text_embed'].to(self.device)
                _, embeddings_1 = self.ae(image_1)
                _, embeddings_2 = self.ae(image_2)
                all_embeddings_1.append(embeddings_1)
                all_embeddings_2.append(embeddings_2)
                all_embeddings_relation.append(relation_emb)
        all_embeddings_1 = torch.cat(all_embeddings_1, dim=0)
        all_embeddings_2 = torch.cat(all_embeddings_2, dim=0)
        all_embeddings_relation = torch.cat(all_embeddings_relation, dim=0)

        # train_data is a dic to store all text1, text2, textR
        # train_data = {'text1': all_embeddings_1, 'text2': all_embeddings_2, 'textR': all_embeddings_relation}
        
        train_data = load_dataset('csv',  data_files=f"{self.current_dir}/cache/{self.task_name}/train_embeddings.csv")['train']
        # get all text 1
        if self.ele_number > 0:
            all_label1 = [i for i in train_data['text1'][:self.ele_number]]
            all_label2 = [ i for i in train_data['text2'][:self.ele_number]]
        else:
            all_label1 = [i for i in train_data['text1']]
            all_label2 = [ i for i in train_data['text2']]
            self.all_entity_labels = all_label1 + all_label2

        all_relations_in_order = [i for i in train_data['textR']]
        # get the entity to entity_index list 
        self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity(train_data)
        
        
        # get the relations  occurrence index
        all_relation_occurrence_index = {}
        unique_relations  = 0
        for i in all_relations_in_order:
            if i not in all_relation_occurrence_index:
                all_relation_occurrence_index[i] = unique_relations
                unique_relations += 1

        self.all_relation_index = torch.tensor([i for i in range(len(all_relations_in_order))]).to(self.device)
        
        # get the unique relation and use the first relation index as the index 
        self.all_relation_mapping = []
        for i in train_data['textR']:
            self.all_relation_mapping.append(all_relation_occurrence_index[i])
        self.all_relation_mapping = torch.tensor(self.all_relation_mapping).to(self.device)
        
        
        self.all_first_entity = [self.return_representative_index(i) for i in range(len(train_data['text1']))]
        self.all_first_entity = torch.tensor(self.all_first_entity).unsqueeze(0).to(self.device)
        
        self.all_second_entity = [self.return_representative_index(i) for i in range(len(train_data['text1']), 2*len(train_data['text1']) )]
        self.all_second_entity = torch.tensor(self.all_second_entity).unsqueeze(0).to(self.device)

        self.all_relation_index = self.all_relation_mapping[self.all_relation_index].unsqueeze(0)
        self.all_fact_index = torch.concat([self.all_first_entity, self.all_second_entity, self.all_relation_index], dim=0).T
        
        self.unique_relation_index = torch.unique(self.all_relation_index, sorted=False).unsqueeze(0)
        self.unique_relation_index.expand(self.number_variable_terms, -1)
        self.unique_relation_index = self.unique_relation_index.unsqueeze(2)
        
        relation_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/relation_token.hf")
        target_atoms_index = []
        ini_index = 0
        for item in train_data:
            if item['textR'] == self.target_relation:
                target_atoms_index.append(ini_index)
            ini_index = ini_index + 1
        self.tar_predicate_index = None
        number_relations = unique_relations
        self.all_relations = relation_data['text']
        print('All relations:', self.all_relations) 

        # get predicate arity 
        predicate_arity = self.check_arity_predicate(train_data)
        print('Predicate arity:', predicate_arity)


        
        # get the arity of the target predicate 
        self.target_predicate_arity = predicate_arity[self.target_relation]
        
        # get all atom arrangement 
        self.all_unground_atoms = []
        for arrage in self.variable_perturbations_list:
            for item in self.all_relations:
                self.all_unground_atoms.append(f"{item}[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")
        target_index = self.all_unground_atoms.index(f'{self.target_relation}[{self.target_variable_arrange}]')
        for item in range(number_relations):
            if relation_data[item]['text'] == self.target_relation:
                self.tar_predicate_index = item
                break
        self.target_atom_index = number_relations * self.get_index_X_Y + self.tar_predicate_index
        # self.target_atom_index = self.get_index_X_Y * self.number_variable_terms + self.target_atom_index
        assert self.target_atom_index == target_index, f"Target atom index {self.target_atom_index} is not equal to the target index {target_index}"
        
        # get the valid list and valid unground atoms 
        invalid_atom_index = []
        valid_atom_index = []
        for ele in range(len(self.all_unground_atoms)):
            item = self.all_unground_atoms[ele]
            predicate = item.split('[')[0]
            first_variable = item.split('[')[1].split('@')[0]
            second_variable = item.split('[')[1].split('@')[1].split(']')[0]
            if first_variable == second_variable:
                atom_variable = 1
            else:
                atom_variable = 2
            if (predicate_arity[predicate] == 2 and atom_variable == 1) or (predicate_arity[predicate] == 1 and atom_variable == 2):
                invalid_atom_index.append(ele)
            else:
                valid_atom_index.append(ele)
        self.valid_atom_index = valid_atom_index
        self.invalid_atom_index = invalid_atom_index
        self.valid_unground_atoms = []
        for ele in valid_atom_index:
            self.valid_unground_atoms.append(self.all_unground_atoms[ele])
        print('Valid unground atoms:', self.valid_unground_atoms)
        
        # ! find and update the target atom index in the valid unground atoms
        self.target_atom_index = self.valid_unground_atoms.index(f'{self.target_relation}[{self.target_variable_arrange}]')
        
        
        # load the positive and negative data and an random index 
        # train_data_E = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_E_{self.tokenize_length}.pt")
        # train_data_R = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_R_{self.tokenize_length}.pt")
        # train_data_L = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_L_{self.tokenize_length}.pt")

        train_data_E = all_embeddings_1
        train_data_L = all_embeddings_2
        train_data_R = all_embeddings_relation
        
        if self.ele_number > 0:    
            selected_train_E = train_data_E[:self.ele_number].to(self.device)
            selected_train_L = train_data_L[:self.ele_number].to(self.device)
            selected_train_R = train_data_R[:self.ele_number].to(self.device)
        else:
            selected_train_E = train_data_E.to(self.device)
            selected_train_L = train_data_L.to(self.device)
            selected_train_R = train_data_R.to(self.device)
        
        # prepare build all KB with bert embedding 
        #! the arrange of embeddings is Entity1 Entity2 Relation
        all_facts_bert_embeddings = torch.cat([selected_train_E, selected_train_L, selected_train_R], dim=1) # get all true index of fact
        
        all_obj_bert_embeddings = torch.cat([selected_train_E, selected_train_L], dim=0) # get all true index of object
        
        # Unique the relation bert embeddings and keep the first occurrence as the index 
        # Step 1: Track unique rows and map them to indices
        seen = {}
        unique_rows = []
        inverse_indices = []

        for row in selected_train_R:
            row_tuple = tuple(row.tolist())
            if row_tuple not in seen:
                seen[row_tuple] = len(seen)  # assign new index
                unique_rows.append(row)
            inverse_indices.append(seen[row_tuple])

        # Step 2: Convert to tensors
        unique_tensor = torch.stack(unique_rows)               # (num_unique, D)
        inverse_tensor = torch.tensor(inverse_indices)         # (N,)

        print("Original tensor:\n", selected_train_R)
        print("Unique rows (first occurrence order):\n", unique_tensor)
        print("Inverse indices:\n", inverse_tensor)
        
        # all_relation_bert_embedding [44,768]
        selected_r_bert_embeddings = unique_tensor.unsqueeze(0)
        original_shape_selected_r_bert_embeddings = unique_tensor
        selected_r_bert_embeddings  = selected_r_bert_embeddings.expand(self.number_variable_terms, -1, -1)
        selected_r_bert_embeddings = selected_r_bert_embeddings.unsqueeze(2)
        
        
        # define the data 
        if self.data_format == 'kg':
            if self.substitution_method == 'random' or self.number_variable > 3:
                tuple_data = BatchZ(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity, number_variable = self.number_variable)
                substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True)
            elif self.substitution_method == 'chain_random':
                tuple_data = BatchChain(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity,index_to_entity=self.index_to_entity, entity_to_entityindex=self.entity_entityindex, entity_to_index = self.entity_index)
                substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=self.batch_size, shuffle=True)
            elif self.substitution_method == 'all':
                tuple_data = AllSubstitution(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity,index_to_entity=self.index_to_entity, entity_to_index=self.entity_index)
                substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=self.random_negative, shuffle=True)
        # load pretrain weights 
        head_model = FactHead(200)
        #! load pretrain weights
        if self.load_pretrain_head:
            checkpoint = torch.load(self.model_path_predicate_path, map_location='cpu')
            head_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            pass
        head_model.to(self.device)

        # Load neural logic rule learning model 
        self.logic_model = DeepRuleLayer(in_size=len(self.valid_unground_atoms)-1, rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=self.rule_path, t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=self.target_arity,time_stamp='')
        self.logic_model.to(self.device)
        
        # Define the differentiable k-means module for image task
        if self.data_format == 'image_ae':
            self.cluster = DkmCompGraph(ae_specs=self.bert_dim_length, n_clusters=self.n_clusters, val_lambda=1, input_size=None, alpha=self.alpha,device=self.device, mode='without_vae')
            # with all entity embeddings from image for initialization all_obj_bert_embeddings, compute the centeriod with trasitional kmeans 
            all_obj_bert_embeddings_numpy = all_obj_bert_embeddings.cpu().numpy()
            print('[pretraining done]')
            kmeans_model = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(all_obj_bert_embeddings_numpy)
            print('[kmeans done]')
            print('[K-Means prediction for embeddings]')
            all_obj_symbolic_representation = kmeans_model.predict(all_obj_bert_embeddings_numpy)
            print(all_obj_symbolic_representation)
            print('[Labels]')
            print(self.all_entity_labels)
            if type(self.all_entity_labels[0]) == str:
                # Create a mapping: string → unique int
                str_to_int = {s: i for i, s in enumerate(sorted(set(self.all_entity_labels)))}
                # Map each string to its corresponding int
                int_list = [str_to_int[s] for s in self.all_entity_labels]
                print(int_list)  # e.g., [0, 1, 0, 2, 1]
                self.all_entity_labels_index = int_list
            else:
                self.all_entity_labels_index = self.all_entity_labels
            acc_initial = accuracy_score(kmeans_model.labels_, self.all_entity_labels_index)
            self.plot_cluster_label_confusion(y_true=self.all_entity_labels_index,y_pred=kmeans_model.labels_)
            print(f'[Accuracy of initial K-Means]')
            print(acc_initial)
            self.cluster.cluster_rep = torch.nn.Parameter(torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32, device=self.device), requires_grad=True)


        # Define parameter groups
        if self.data_format == 'image_ae':
            parameter_groups = [
                {'params': self.logic_model.parameters(), 'lr': self.lr_rule_model},  # Lower learning rate for the first layer
                {'params': head_model.parameters(), 'lr': self.lr_predicate_model},  # Higher learning rate for the second layer
                {'params': self.cluster.parameters(), 'lr': self.lr_dkm}  # Higher learning rate for the second layer
            ]
        else:
            parameter_groups = [
                {'params': self.logic_model.parameters(), 'lr': self.lr_rule_model},  # Lower learning rate for the first layer
                {'params': head_model.parameters(), 'lr': self.lr_predicate_model}  # Higher learning rate for the second layer
            ]
        logic_optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)

        # load pre rules 
        try:
            with open(self.rule_path+'.pk', "rb") as f:
                rule_set = pickle.load(f)
                f.close()
        except FileNotFoundError:
            print(f"Rule file {self.rule_path+'.pk'} not found. Starting with an empty rule set.")
            rule_set = None

        # start training
        self.logic_model.train()
        head_model.train()
        infer_time = 0
        for single_epoch in range(self.epoch):
            total_loss_rule = 0
            total_loss = 0
            total_dkm_loss = 0
            total_body_loss = 0
            if self.data_format == 'image' or self.data_format == 'image_ae':
                self.cluster.train()
            #todo change the dataset for each epoch when learning from images, adjust the batch symbolic representations 
            if self.data_format == 'image' or self.data_format == 'image_ae':
                if self.substitution_method == 'random' or self.number_variable >= 3:
                    all_obj_bert_embeddings, all_obj_symbolic_representation, kmeans_loss = self.cluster(all_obj_bert_embeddings)
                    # update the knowledge base 
                    all_facts_bert_embeddings = self.update_all_fact_embedding_image(all_obj_bert_embeddings, selected_train_R)

                    tuple_data = BatchZ(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity, number_variable = self.number_variable)
                    substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True)

                else:
                    self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity_cluster(all_obj_symbolic_representation, train_data_E.shape[0])
                    tuple_data = BatchChain(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity,index_to_entity=self.index_to_entity, entity_to_entityindex=self.entity_entityindex, entity_to_index = self.entity_index)
                    substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True)
            
            for index in tqdm.tqdm(substation_data):
                if self.substitution_method == 'all':
                    index = torch.stack(index)
                    index = index.transpose(0,1)
                logic_optimizer.zero_grad()
                index = index.to(self.device).int()
                
                # when multiple dimension, turn them into binary siz 
                index = index.reshape(-1, index.shape[-1])
                
                # index = index.squeeze(0)
                # transform the index                 
                number_instance = index.shape[0]

                # # ! We only consider the X Y Z three variables in this stage
                # obtain the substitution for all atoms in body, get the all variable perturbation and return the substitutions based on X, Y, and Z variable
                all_row = torch.arange(0, index.shape[0], device=self.device).unsqueeze(1).expand(-1, self.variable_perturbations.shape[1])
                all_row = all_row.unsqueeze(1).expand(-1, self.variable_perturbations.shape[0],-1)
                variable_perturbations_variance = self.variable_perturbations.unsqueeze(0).expand(index.shape[0], -1, -1)
                perturbations = index[all_row, variable_perturbations_variance]
                
                # todo using perturbations to get the labels
                # combine the relation index to the perturbations 
                
                if self.data_format == 'kg':
                    pertubations_relations = perturbations.unsqueeze(2)
                    pertubations_relations_expand = pertubations_relations.expand(-1, -1, number_relations, -1)
                    unique_relation_index = self.unique_relation_index.expand(pertubations_relations_expand.shape[1], -1, -1)
                    unique_relation_index = unique_relation_index.unsqueeze(0).expand(pertubations_relations_expand.shape[0], -1, -1, -1)
                    combine_facts = torch.cat([pertubations_relations_expand, unique_relation_index], dim=3)
                    combine_facts = combine_facts.reshape(number_instance, -1, 3)
                    combine_facts_ready_check = combine_facts.reshape(-1, 3)
                    # check the occurrence
                    combine_facts_ready_check = combine_facts_ready_check.unsqueeze(1)
                    all_facts_var = self.all_fact_index.unsqueeze(0)
                    labels = (combine_facts_ready_check==all_facts_var).all(dim=2).any(dim=1)
                    labels = labels.reshape(number_instance, -1).float()
                    labels_body = labels
                
                elif self.data_format == 'image_ae':
                    # first find labels for all perturbation 
                    # modify all_obj_bert_embeddings to centeriod embeddings
                    all_obj_bert_embeddings_centeriods, all_obj_symbolic_representation, kmeans_loss = self.cluster(all_obj_bert_embeddings)
                    # encoder = Encoder(768, 20)
                    # all_obj_bert_embeddings = encoder(all_obj_bert_embeddings)
                    
                    # update the knowledge base 
                    all_facts_bert_embeddings = self.update_all_fact_embedding_image(all_obj_bert_embeddings_centeriods, selected_train_R)
                    
                    
                    
                    #! Based on the all body predicates and entity pairs, generate the labels 
                    perturbations_bert_embeddings = all_obj_bert_embeddings_centeriods[perturbations]
                    # assign the clustering centroid to the entity and using the embedding of the centroid as the embedding of entities
                    perturbations_bert_embeddings = perturbations_bert_embeddings.unsqueeze(2)
                    expanded_per = perturbations_bert_embeddings.expand(-1, -1, number_relations, -1, -1)                
                    selected_r_bert_embeddings_var = selected_r_bert_embeddings.unsqueeze(0).expand(expanded_per.shape[0], -1, -1, -1, -1)
                    triples = torch.cat([expanded_per, selected_r_bert_embeddings_var], dim=3)
                    triples = triples.reshape(number_instance, -1, 3, self.bert_dim_length)
                    triples = triples.reshape(-1, 3, self.bert_dim_length)
                    first_shape = triples.shape[0]
                    triples = triples.reshape(first_shape,-1)
                    # check the occurrence 
                    triples = triples.unsqueeze(1)
                    all_facts_bert_embeddings_var = all_facts_bert_embeddings.unsqueeze(0)
                    labels_body = self.batched_row_existence(triples, all_facts_bert_embeddings_var)
                    # labels_body = (triples==all_facts_bert_embeddings_var).all(dim=2).any(dim=1)
                    labels_body = labels_body.reshape(number_instance, -1).float()
                
                
                # assert torch.allclose(labels, labels_body, rtol=1e-05, atol=1e-08), "Tensors are not equal"
                # get the head embeddings for all perturbations 
                # todo the number of false head is missing 
                # todo all body are all zero because the limited z 
                
                # get the head embedding of the entity 
                if self.data_format == 'image_ae':
                    #! for image task, we need to use the cluster centeriod as the entity embedding 
                    selected_train_E_head_embeddings, clusters1, l1 = self.cluster(selected_train_E)
                    selected_train_L_head_embeddings, clusters2, l2 = self.cluster(selected_train_L)
                    selected_train_E_head_embeddings = head_model.generate_embedding(selected_train_E_head_embeddings)
                    selected_train_L_head_embeddings = head_model.generate_embedding(selected_train_L_head_embeddings)
                else:
                    selected_train_E_head_embeddings = head_model.generate_embedding(selected_train_E)
                    selected_train_L_head_embeddings = head_model.generate_embedding(selected_train_L)
                selected_r_head_embeddings = head_model.generate_embedding(original_shape_selected_r_bert_embeddings)
                all_objects_head_embedding = torch.cat([selected_train_E_head_embeddings, selected_train_L_head_embeddings], dim=0).to(self.device)
                selected_r_head_embeddings = selected_r_head_embeddings.unsqueeze(0)
                selected_r_head_embeddings  = selected_r_head_embeddings.expand(self.number_variable_terms, -1, -1)
                selected_r_head_embeddings = selected_r_head_embeddings.unsqueeze(2)
                
                perturbations_embeddings = all_objects_head_embedding[perturbations.int()]
                perturbations_embeddings = perturbations_embeddings.unsqueeze(2)
                expanded_per = perturbations_embeddings.expand(-1, -1, number_relations, -1, -1)
                all_r_embeddings_variance = selected_r_head_embeddings.unsqueeze(0).expand(expanded_per.shape[0], -1, -1, -1, -1)
                # ! Combine the relation to all entities 
                triples = torch.cat([expanded_per, all_r_embeddings_variance], dim=3)    
                triples = triples.reshape(number_instance, -1, 3, self.final_dim) # R_0(X,Y), R_1(X,Y), ...,R_n(X,Y) ,R_0(Y,X),.., R_n(Y,X), ... X (e1, e2, r) X embedding length 
                triples = triples.reshape(-1, 3, self.final_dim)
                
                # get the distance between the entity and relation
                e_hidden_states = triples[:,0,:]
                r_hidden_states = triples[:,2,:]
                l_hidden_states = triples[:,1,:]
                ground_truth_values = head_model.get_possible(e1 = e_hidden_states,  embed_r =  r_hidden_states, e2 = l_hidden_states)
                ground_truth_values = ground_truth_values.reshape(number_instance, -1)

                # ground_truth_values = ground_truth_values.reshape(-1)
                # labels_body = labels_body.reshape(-1)
                
                # get valid atom labels and ground truth values
                if self.open_neural_predicate == True:
                    ground_truth_values = ground_truth_values[:,self.valid_atom_index]
                    labels_body = labels_body[:,self.valid_atom_index]
                    body_loss = torch.nn.BCELoss()(ground_truth_values, labels_body) # can use Cross Entropy Loss
                else:
                    # todo the valid atom index can be updated based on the index from clustering 
                    labels_body = labels_body[:,self.valid_atom_index]
                    ground_truth_values = labels_body
                    body_loss = 0
                head_predicate_labels_predicated = ground_truth_values[:,self.target_atom_index].unsqueeze(1)
                body_predicate_labels_predicated = torch.cat([ground_truth_values[:,:self.target_atom_index], ground_truth_values[:,self.target_atom_index+1:]],dim=1)

                # using logic model and body atoms to predict the head predicate
                predicated = self.logic_model(body_predicate_labels_predicated)
                loss_rule = torch.nn.MSELoss()(predicated, head_predicate_labels_predicated)

                #loss for dkm 
                if self.data_format == 'image_ae' and self.open_neural_predicate == True:
                    dkm_loss = l1 + l2 + kmeans_loss
                elif self.data_format == 'image_ae' and self.open_neural_predicate == False:
                    dkm_loss = kmeans_loss
                else: 
                    dkm_loss = 0
                
                # all loss from logic rule loss, body predicate, and [deep k-means]
                if self.open_neural_predicate == True:
                    loss = 0.5*loss_rule + 0.5*body_loss + 0.5 * dkm_loss
                else:
                    # loss = 0.5*loss_rule + 0.5 * dkm_loss
                    loss = loss_rule + self.lambda_dkm * dkm_loss
                

                loss.backward(retain_graph=True)
                # todo define the loss from neural predicate here 
                logic_optimizer.step()
                total_loss_rule = total_loss_rule + loss_rule.item()
                total_loss += loss.item()
                if self.open_neural_predicate == True:
                    total_loss += body_loss.item()
                else:
                    total_loss += 0
                if self.data_format == 'image_ae':
                    total_dkm_loss += dkm_loss.item()
                else:
                    total_dkm_loss += 0

            # if the input is the image, build the knowledge graphs based on the clustering centroid as the entities and the existing relations as the predicates
            dkm_acc = 0
            if self.data_format == 'image_ae':
                dkm_acc = self.build_kg_based_on_cluster(selected_train_E, selected_train_L, all_relations_in_order)
                # build on test data
                # infer_time = self.infer(infer_time)
                

            total_loss_rule = total_loss_rule / len(substation_data)
            total_loss = total_loss / len(substation_data)
            total_dkm_loss = total_dkm_loss / len(substation_data)
            total_body_loss = total_body_loss / len(substation_data)
            print(f"Epoch: {single_epoch}, Loss: {total_loss}, Body Loss: {total_body_loss}, Rule Loss: {total_loss_rule}, DKM Loss: {total_dkm_loss}, DKM acc, {dkm_acc}")
            self.writer.add_scalar("Loss/train", total_loss, single_epoch)
            self.writer.add_scalar("Loss/train_Logic", total_loss_rule, single_epoch)
            self.writer.add_scalar("Loss/body_loss", total_body_loss, single_epoch)
            self.writer.add_scalar("Loss/dkm_loss", total_dkm_loss, single_epoch)
            self.writer.add_scalar("ACC/dkm_acc", dkm_acc, single_epoch)
            self.writer.flush()
            
            # compute the acc on the knowledge graphs 
            train_acc, rule_set = self.check_acc(rule_set)
            # save rule set 
            with open(self.rule_path+'.pk', "wb") as f:
                pickle.dump(rule_set, f)
                f.close()
            train_precision = train_acc['precision']
            train_recall = train_acc['recall']
            if train_recall >= self.stop_recall:
                break
        self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
        print("[Training Finished]")
        print("[Rules]")
        print(rule_set)
        print(f"[Precision]")
        print(train_precision)
        print(f"[Recall]")
        print(train_recall)
        return 0 
    
    def predicate_invention_autoencoder(self, inner_batch = 2):
        '''
        Read all entities and process and propositionalalization and rule learning task together.
        Train distance and logic model together.
        BERT embedding is fix but head embeddings is trainable.
        '''
        self.number_variable = self.n_clusters
        # consider the binary predicate and unary predicate at the same time 
        if self.body_arity == '12':
            self.variable_perturbations_list = list(itertools.combinations(range(self.number_variable), 2)) + [(x,x) for x in range(self.number_variable)]
        elif self.body_arity == '1':
            self.variable_perturbations_list = [(x,x) for x in range(self.number_variable)]
        elif self.body_arity == '2':
            self.variable_perturbations_list = list(itertools.combinations(range(self.number_variable), 2))
        else:
            raise ValueError("The body arity should be 1, 2, or 12")
        self.variable_perturbations = torch.tensor(self.variable_perturbations_list, device=self.device)
        self.all_unground_atoms = []
        self.all_relations = [''] # placeholder for the invented predicate. For each tuple of variable, only one invented predicate will be created
        for arrage in self.variable_perturbations_list:
            for item in self.all_relations:
                self.all_unground_atoms.append(f"{item}[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")
        self.valid_unground_atoms = self.all_unground_atoms
        self.inner_batch = inner_batch
        
        # load the positive and negative data and an random index 
        # train_data_pos = torch.load(f"{self.current_dir}/cache/{self.task_name}/positive_images_data_train.pt")
        # train_data_neg = torch.load(f"{self.current_dir}/cache/{self.task_name}/negative_images_data_train.pt")
        # test_data_pos = torch.load(f"{self.current_dir}/cache/{self.task_name}/positive_images_data_test.pt")
        # test_data_neg = torch.load(f"{self.current_dir}/cache/{self.task_name}/negative_images_data_test.pt")
        # all_obj_bert_embeddings = torch.concat([train_data_pos, train_data_neg], dim=0).to(self.device) # get all true index of object
        
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # get the positive objects uner {self.current_dir}/train/true/cropped_objects/00* but expect result_preview.png
        positive_dataset = MultiFolderDataset(parent_dir=f"{self.current_dir}/dat-kandinsky-patterns/{self.task_name}/train/true/cropped_objects", transform=transform)
        
        # get the negatice objects uner {self.current_dir}/train/false/cropped_objects/00* but expect result_preview.png
        negative_dataset = MultiFolderDataset(parent_dir=f"{self.current_dir}/dat-kandinsky-patterns/{self.task_name}/train/false/cropped_objects", transform=transform)
        
        positive_dataset_test = MultiFolderDataset(parent_dir=f"{self.current_dir}/dat-kandinsky-patterns/{self.task_name}/test/true/cropped_objects", transform=transform)
        negative_dataset_test = MultiFolderDataset(parent_dir=f"{self.current_dir}/dat-kandinsky-patterns/{self.task_name}/test/false/cropped_objects", transform=transform)
        
        
        # Combine positive and negative datasets
        combined_dataset = torch.utils.data.ConcatDataset([positive_dataset, negative_dataset])
        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=268, shuffle=True)
        # Pretrain AutoEncoder
        self.ae = ConvAutoencoder().to(self.device)
        ae_optimizer = torch.optim.Adam(self.ae.parameters(), lr=1e-3)
        # autoencoder training time 
        pre_train_ae_num_epochs = self.pre_train_ae_num_epochs
        for epoch in range(pre_train_ae_num_epochs):
            for images, _ in dataloader:
                images = images.to(self.device)
                reconstructed, embeddings = self.ae(images)
                loss = nn.MSELoss()(reconstructed, images)
                ae_optimizer.zero_grad()
                loss.backward()
                ae_optimizer.step()
            print(f"Epoch [{epoch+1}/{pre_train_ae_num_epochs}], Loss: {loss.item():.4f}")
        
        # Get embeddings from the encoder part of the autoencoder
        # all images into their corresponding embeddings
        all_obj_bert_embeddings = []
        self.index_to_image = []
        with torch.no_grad():
            for images, paths in dataloader:
                images = images.to(self.device)
                _, embeddings = self.ae(images)
                all_obj_bert_embeddings.append(embeddings)
                self.index_to_image.extend(paths)
        all_obj_bert_embeddings = torch.cat(all_obj_bert_embeddings, dim=0)
        
        
        
        
        
        
        # load the index to image file 
        # with open(f"{self.current_dir}/cache/{self.task_name}/index_to_image_train.pkl", 'rb') as f:
            # self.index_to_image_2 = pickle.load(f)
            # f.close()
        

                
            
            
        # 1. Load neural logic rule learning model 
        self.logic_model = DeepRuleLayer(in_size=len(self.all_unground_atoms), rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=self.rule_path, t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=self.target_arity,time_stamp='')
        self.logic_model.to(self.device)
        
        
        
        # 2. Define the differentiable k-means module for image task
        self.cluster = DkmCompGraph(ae_specs=self.bert_dim_length, n_clusters=self.n_clusters, val_lambda=1, input_size=None, alpha=self.alpha,device=self.device, mode='without_vae')
        # with all entity embeddings from image for initialization all_obj_bert_embeddings, compute the centeriod with trasitional kmeans 
        # reshape embedding into -1 768
        all_obj_bert_embeddings_numpy = all_obj_bert_embeddings.view(-1, self.bert_dim_length).cpu().numpy()  # Reshape to (-1, 768)
        print('[pretraining done]')
        kmeans_model = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(all_obj_bert_embeddings_numpy)
        print('[kmeans done]')
        print('[K-Means prediction for embeddings]')
        all_obj_symbolic_representation = kmeans_model.predict(all_obj_bert_embeddings_numpy)
        print(all_obj_symbolic_representation)
        self.plot_cluster(all_obj_bert_embeddings_numpy, self.index_to_image, cluster_labels = all_obj_symbolic_representation, addition_inputs = 'init_')

            
        # print(f'[Accuracy of initial K-Means]')
        # print(acc_initial)
        self.cluster.cluster_rep = torch.nn.Parameter(torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32, device=self.device), requires_grad=True)

        # Define parameter groups
        parameter_groups = [
            {'params': self.logic_model.parameters(), 'lr': self.lr_rule_model},  # Lower learning rate for the first layer
            {'params': self.cluster.parameters(), 'lr': self.lr_dkm},  # Higher learning rate for the second layer
            {'params': self.ae.parameters(), 'lr': 0}  # Learning rate for autoencoder
        ]
        logic_optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)


        # start training
        self.logic_model.train()
        rule_set = None
        infer_time = 0
        for single_epoch in range(self.epoch):
            total_loss_rule = 0
            total_loss = 0
            total_ae_loss = 0
            total_dkm_loss = 0
            total_body_loss = 0

            self.cluster.train()
            #todo change the dataset for each epoch when learning from images, adjust the batch symbolic representations 
            # self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity_cluster(all_obj_symbolic_representation, train_data_E.shape[0])
            
            #use the batch_pi_autoencoder as dataloader 
            tuple_data = Batch_PI_auto_encoder(positive_dataset, negative_dataset, inner_batch = self.inner_batch)
            # tuple_data = Batch_PI(train_data_pos, train_data_neg, inner_batch = self.inner_batch)
            substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=self.batch_size, shuffle=True)
            
            for original_images, label in tqdm.tqdm(substation_data):
                original_images= original_images.to(self.device)
                logic_optimizer.zero_grad()
                entities_number = original_images.shape[1]
                # get the continues four images into their embeddings generated from autoencoder 
                original_images = original_images.view(-1, 3, 224, 224)  # Reshape to (batch_size * inner_batch, 3, 224, 224)
                reconstructed_images, index = self.ae(original_images)
                # recovde
                index = index.view(-1, entities_number, self.bert_dim_length)  # Reshape back to (batch_size, inner_batch, embedding_dim)
                label  = label.to(self.device).float()
                # index = index.squeeze(0)
                # transform the index        
                #! take special attention why 3 is chosen 
                index_inedx = torch.randint(0, entities_number, (self.inner_batch, 3)).to(self.device) # current the v
                # select on each binary matrix 
                x = index 
                idx = index_inedx
                # Expand idx for batch dimension
                # [28, 3] -> [1024, 28, 3]
                idx_exp = idx.unsqueeze(0).expand(x.size(0), -1, -1)

                # Build batch indices [1024, 28, 3]
                batch_idx = torch.arange(x.size(0)).view(-1, 1, 1).expand_as(idx_exp)

                # Use advanced indexing
                out = x[batch_idx, idx_exp]   # [1024, 28, 3, 768]
                
                # index = index[index_inedx] 
                index = out 
                first_shape = index.shape[0]
                all_data = index.reshape(-1, self.bert_dim_length)
                
                # get the symbolic representation for the embeddings 
                all_obj_bert_embeddings_centeriods, all_obj_symbolic_representation_diff, kmeans_loss = self.cluster(all_data)
                # all_data_numpy = all_data.cpu().numpy()
                # all_obj_symbolic_representation = torch.tensor(kmeans_model.predict(all_data_numpy), device=self.device)
                # print(all_obj_symbolic_representation)
                # print(all_obj_symbolic_representation_diff)
                all_obj_symbolic_representation = all_obj_symbolic_representation_diff.reshape(first_shape, self.inner_batch,-1)
                # sort each element in all_obj_symbolic_representation in the second dimension
                # get all combinations of two elements in each row
                n = all_obj_symbolic_representation.size(-1)
                # Get upper-triangle indices (combinations of 2 elements without repetition)
                idx = torch.triu_indices(n, n, offset=0)
                # Get the two elements from each row using the indices
                comb1 = all_obj_symbolic_representation[:, :, idx[0]]  # Shape: (batch_size, num_combinations)
                comb2 = all_obj_symbolic_representation[:, :, idx[1]]  # Same
                # Stack along a new dimension to form pairs
                binary_facts = torch.stack((comb1, comb2), dim=3)  # Shape: (batch_size, num_combinations, 2)
                # sort in the third dimension
                all_facts = torch.sort(binary_facts, dim=-1)[0]
                all_facts_unique = all_facts.unique(dim=2)  # Remove duplicate pairs across the batch
                
                # check the template is in the facts or not 
                # Step 1: Reshape A to (1, n, d) and B to (m, n, d)
                # We'll compare each row in A to all rows in B across axis
                A = self.variable_perturbations  # shape (1, n, d)
                B = all_facts_unique  # shape (m, n, d)

                A_exp = A.unsqueeze(0).unsqueeze(2)      # [1, 6, 1, 2]
                B_exp = B.unsqueeze(2)                   # [3, 1, 3, 2]

                # Compare each A[i] with all elements in each B[b]
                matches = (A_exp == B_exp).all(dim=-1)   # [1, 6, 3]

                # Check if A[i] exists in any row in B[b]
                exists = matches.any(dim=-1).unsqueeze(-1)  # [1, 6, 1]

                # print(exists)
                
                # using logic model and body atoms to predict the head predicate
                body_predicate_labels_predicated = exists.reshape(first_shape, self.inner_batch, -1).float()
                body_predicate_labels_predicated = body_predicate_labels_predicated.reshape(first_shape*self.inner_batch, -1)  # Reshape to match the label shape
                predicated = self.logic_model(body_predicate_labels_predicated)
                label = label.unsqueeze(1).repeat(1, self.inner_batch)  # Repeat label to match the batch size
                label = label.reshape(first_shape*self.inner_batch, -1)  # Reshape to match the predicated shape
                loss_rule = torch.nn.MSELoss()(predicated, label)
                loss_rule = loss_rule.float()
                
                # autoencode loss 
                ae_loss = nn.MSELoss()(reconstructed_images, original_images)
                
                # all loss from logic rule loss, body predicate, and [deep k-means]
                loss = 0.5*loss_rule + 0.5 * kmeans_loss + 0.1 * ae_loss
                
                loss.backward(retain_graph=True)
                logic_optimizer.step()
                total_loss_rule += loss_rule.item()
                total_dkm_loss += kmeans_loss.item()
                total_ae_loss += ae_loss.item()
                total_loss += loss.item()


            # if the input is the image, build the knowledge graphs based on the clustering centroid as the entities and the existing relations as the predicates
            dkm_acc = 0

            # build on test data
            # infer_time = self.infer(infer_time)
            # update all obj embeddigns from autoencoder 
            all_obj_bert_embeddings = []
            self.index_to_image = []
            with torch.no_grad():
                for images, paths in dataloader:
                    images = images.to(self.device)
                    _, embeddings = self.ae(images)
                    all_obj_bert_embeddings.append(embeddings)
                    self.index_to_image.extend(paths)
            all_obj_bert_embeddings = torch.cat(all_obj_bert_embeddings, dim=0)
            
            
            self.plot_cluster(all_obj_bert_embeddings, self.index_to_image)
            total_loss_rule = total_loss_rule / len(substation_data)
            total_dkm_loss = total_dkm_loss / len(substation_data)
            total_ae_loss = total_ae_loss / len(substation_data)
            total_loss = total_loss / len(substation_data)
            print(f"Epoch: {single_epoch}, Loss: {total_loss}, Body Loss: {total_body_loss}, Rule Loss: {total_loss_rule}, DKM Loss: {total_dkm_loss}, DKM acc, {dkm_acc}, AE Loss: {total_ae_loss}")
            self.writer.add_scalar("Loss/train", total_loss, single_epoch)
            self.writer.add_scalar("Loss/train_Logic", total_loss_rule, single_epoch)
            self.writer.add_scalar("Loss/body_loss", total_body_loss, single_epoch)
            self.writer.add_scalar("Loss/dkm_loss", total_dkm_loss, single_epoch)
            self.writer.add_scalar("Loss/ae_loss", total_ae_loss, single_epoch)
            self.writer.add_scalar("ACC/dkm_acc", dkm_acc, single_epoch)
            self.writer.flush()
            
            # compute the acc on the knowledge graphs 
            # todo consider how to add it 
            # get the iamges entities from test data 

            dataloader_test_positive = torch.utils.data.DataLoader(positive_dataset_test, batch_size=32, shuffle=False)
            dataloader_test_negative = torch.utils.data.DataLoader(negative_dataset_test, batch_size=32, shuffle=False)
            all_obj_bert_embeddings_test_pos = []
            with torch.no_grad():
                for images, _ in dataloader_test_positive:
                    images = images.to(self.device)
                    _, embeddings = self.ae(images)
                    all_obj_bert_embeddings_test_pos.append(embeddings)
            all_obj_bert_embeddings_test_pos = torch.cat(all_obj_bert_embeddings_test_pos, dim=0)
            all_obj_bert_embeddings_test_neg = []
            with torch.no_grad():
                for images, _ in dataloader_test_negative:
                    images = images.to(self.device)
                    _, embeddings = self.ae(images)
                    all_obj_bert_embeddings_test_neg.append(embeddings)
            all_obj_bert_embeddings_test_neg = torch.cat(all_obj_bert_embeddings_test_neg, dim=0)
            test_data_neg = all_obj_bert_embeddings_test_neg
            test_data_pos = all_obj_bert_embeddings_test_pos
            
            test_data = self.build_kb_based_on_cluster_predicate_invention(test_negative=test_data_neg, test_positive=test_data_pos, cluster_model=kmeans_model, kmtype='non_differentiable')
            acc, rule_set = self.check_acc_PI(test_data, rule_set)
            # save rule set 
            with open(self.rule_path+'.pk', "wb") as f:
                pickle.dump(rule_set, f)
                f.close()
            # todo consider how to add it 
            precision = acc['precision']
            recall = acc['recall']
            if recall >= self.stop_recall:
                break
        self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
        print("[Training Finished]")
        print("[Rules]")
        print(rule_set)
        print(f"Test Precision")
        print(precision)
        print(f"Test Recall")
        print(recall)
        print("Test Acc")
        print(acc['acc'])
        return 0 
    
    
    def plot_cluster_ppi(self, all_obj_bert_embeddings, top_n = 20, cluster_labels = None, addition_inputs = ''):
        length_facts = int(all_obj_bert_embeddings.shape[0]/2)
        if top_n > length_facts:
            top_n = length_facts
        selected_embeddings  = all_obj_bert_embeddings.reshape(-1,self.bert_dim_length)[:top_n]
        selected_embeddings_2 = all_obj_bert_embeddings.reshape(-1,self.bert_dim_length)[length_facts:top_n+ length_facts]
        selected_embeddings = torch.cat([selected_embeddings, selected_embeddings_2], dim=0)


        selected_images = self.index_to_image[:top_n]
        selected_images_2 = self.index_to_image[length_facts:top_n + length_facts]
        selected_images = selected_images + selected_images_2

        if type(cluster_labels) ==  type(None):
            cluster_index  = self.cluster.get_cluster_index(selected_embeddings).detach().cpu().numpy()
        else:
            cluster_index = cluster_labels[:top_n]
        cluster_index = cluster_index.tolist()
        label_to_paths = defaultdict(list)
        for path, label in zip(selected_images, cluster_index):
            label_to_paths[label].append(path)
        # open mnist file 
        with open(f"{self.current_dir}/cache/{self.task_name}/mnist_images.pkl", 'rb') as f:
            mnist_data = pickle.load(f)
            f.close()
        for label, paths in label_to_paths.items():
            n = len(paths)
            plt.figure(figsize=(n * 2, 2))
            for i, path in enumerate(paths):
                img = mnist_data[path].squeeze(0)  # Assuming mnist_data is a dictionary with paths as keys and images as values
                img_np = img.permute(1, 2, 0).numpy()  # convert to (224, 224, 3)
                # Optional: Normalize if needed (e.g., range is not 0–1)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                plt.subplot(1, n, i + 1)
                plt.imshow(img_np, cmap='gray')
                plt.axis('off')
            plt.suptitle(f'Label: {self.mapping[label]}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.current_dir}/cache/{self.task_name}/{addition_inputs}cluster_{self.mapping[label]}.png")
        return 0


    def plot_cluster(self, all_obj_bert_embeddings,index_to_image, top_n = 20, cluster_labels = None, addition_inputs = ''):
        selected_embeddings  = all_obj_bert_embeddings.reshape(-1,self.bert_dim_length)[:top_n]
        
        # index_to_image stores the image paths
        selected_images = index_to_image[:top_n] 
        if type(cluster_labels) ==  type(None):
            cluster_index  = self.cluster.get_cluster_index(selected_embeddings).detach().cpu().numpy()
        else:
            cluster_index = cluster_labels[:top_n]
        cluster_index = cluster_index.tolist()
        # plot the images with same cluster index 
        
        # Group image paths by label
        label_to_paths = defaultdict(list)
        for path, label in zip(selected_images, cluster_index):
            label_to_paths[label].append(path)

        # Plot each group
        for label, paths in label_to_paths.items():
            n = len(paths)
            plt.figure(figsize=(n * 2, 2))
            for i, path in enumerate(paths):
                img = Image.open(path)
                plt.subplot(1, n, i + 1)
                plt.imshow(img)
                plt.axis('off')
            plt.suptitle(f'Label: {self.mapping[label]}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.current_dir}/cache/{self.task_name}/{addition_inputs}cluster_{self.mapping[label]}.png")
        return 0             
    
    def infer(self, infer_time=0):
        # load the positive and negative data and an random index 
        if infer_time == 0:
            self.test_data_E = torch.load(f"{self.current_dir}/cache/{self.task_name}/test_balance_bert_embeddings_E_{self.tokenize_length}.pt").to(self.device)
            self.test_data_L = torch.load(f"{self.current_dir}/cache/{self.task_name}/test_balance_bert_embeddings_L_{self.tokenize_length}.pt").to(self.device)
            self.test_data = load_dataset('csv',  data_files=f"{self.current_dir}/cache/{self.task_name}/test_embeddings.csv")['train']

        all_relations_in_order = [i for i in self.test_data['textR']]
        self.all_obj_label_test = [i for i in self.test_data['text1']] + [i for i in self.test_data['text2']]
        if type(self.all_obj_label_test[0]) == str:
            # Create a mapping: string → unique int
            str_to_int = {s: i for i, s in enumerate(sorted(set(self.all_obj_label_test)))}
            # Map each string to its corresponding int
            int_list = [str_to_int[s] for s in self.all_obj_label_test]
            print(int_list)  # e.g., [0, 1, 0, 2, 1]
            self.all_obj_label_test_index = int_list
        else: 
            self.all_obj_label_test_index = self.all_obj_label_test
        # generate the centriod 
        self.build_kg_based_on_cluster(self.test_data_E, self.test_data_L, all_relations_in_order, 'test')
            
        # output the testing data 
        infer_time += 1
        return infer_time
    
    def check_existence(self, all_obj_symbolic_representation):
        '''
        
        Check whether the tuple from image is in the pre-defined tuple list  
        '''
        batch_size = len(all_obj_symbolic_representation)
        inner_batch = 1
        # check existence 
        all_obj_symbolic_representation = all_obj_symbolic_representation.reshape(batch_size, inner_batch,-1).to(self.device)
        # sort each element in all_obj_symbolic_representation in the second dimension
        # get all combinations of two elements in each row
        n = all_obj_symbolic_representation.size(-1)
        # Get upper-triangle indices (combinations of 2 elements without repetition)
        idx = torch.triu_indices(n, n, offset=0)
        # Get the two elements from each row using the indices
        comb1 = all_obj_symbolic_representation[:, :, idx[0]]  # Shape: (batch_size, num_combinations)
        comb2 = all_obj_symbolic_representation[:, :, idx[1]]  # Same
        # Stack along a new dimension to form pairs
        binary_facts = torch.stack((comb1, comb2), dim=3)  # Shape: (batch_size, num_combinations, 2)
        # sort in the third dimension
        all_facts = torch.sort(binary_facts, dim=-1)[0]
        all_facts_unique = all_facts.unique(dim=2)  # Remove duplicate pairs across the batch
        
        # check the template is in the facts or not 
        # Step 1: Reshape A to (1, n, d) and B to (m, n, d)
        # We'll compare each row in A to all rows in B across axis
        A = self.variable_perturbations  # shape (1, n, d)
        B = all_facts_unique  # shape (m, n, d)

        A_exp = A.unsqueeze(0).unsqueeze(2)      # [1, 6, 1, 2]
        B_exp = B.unsqueeze(2)                   # [3, 1, 3, 2]

        # Compare each A[i] with all elements in each B[b]
        matches = (A_exp == B_exp).all(dim=-1)   # [1, 6, 3]

        # Check if A[i] exists in any row in B[b]
        exists = matches.any(dim=-1).unsqueeze(-1)  # [1, 6, 1]

        # print(exists)
        return exists


        
    def build_kb_based_on_cluster_predicate_invention(self, test_positive, test_negative, cluster_model, kmtype='non_differentiable'):
        '''
        This code is used to check the acc for time series like KB 
        '''
        # train_data_pos = torch.load(f"{self.current_dir}/cache/{self.task_name}/positive_images_data.pt")
        # train_data_neg = torch.load(f"{self.current_dir}/cache/{self.task_name}/negative_images_data.pt")
        number_pos = test_positive.shape[0]
        number_neg = test_negative.shape[0]
        # reshape the train_pos and neg to one dimension
        test_positive = test_positive.reshape(-1, self.bert_dim_length)
        test_negative = test_negative.reshape(-1, self.bert_dim_length)
        # get all cluster index for each data in both positive and negative data
        if kmtype == 'differentiable':
            cluster_model.eval()
            cluster_index_positive = cluster_model.get_cluster_index(test_positive).detach().cpu().numpy()
            cluster_index_negative = cluster_model.get_cluster_index(test_negative).detach().cpu().numpy()
        else:
            cluster_index_positive = cluster_model.predict(test_positive.cpu().numpy())
            cluster_index_negative = cluster_model.predict(test_negative.cpu().numpy())
        
        # reshape cluster into the original instance number with numpy 
        cluster_index_positive = torch.tensor(cluster_index_positive.reshape(number_pos,-1))
        cluster_index_negative = torch.tensor(cluster_index_negative.reshape(number_neg,-1))
        
        exist_pos = self.check_existence(cluster_index_positive).float().detach().cpu().numpy()
        exist_neg = self.check_existence(cluster_index_negative).float().detach().cpu().numpy()
        
        # make pandas dataframe for positive and negative data, the column is self.valid_unground_atoms
        exist_pos = exist_pos.reshape(number_pos, -1).tolist()
        exist_neg = exist_neg.reshape(number_neg, -1).tolist()
        positive_df = pd.DataFrame(exist_pos, columns=self.valid_unground_atoms)
        negative_df = pd.DataFrame(exist_neg, columns=self.valid_unground_atoms)
        positive_df['label'] = 1  # Add label column for positive data
        negative_df['label'] = 0  # Add label column for negative data
        
        # concatenate positive and negative data
        test_data = pd.concat([positive_df, negative_df], ignore_index=True)
        
        # save the data to csv file
        test_data.to_csv(f"{self.current_dir}/cache/{self.task_name}/predicate_invention.csv", index=False)
        return test_data
        
        
    def build_kg_based_on_cluster(self, selected_train_E, selected_train_L, all_relations,data_type=''):
        '''
        This code is used to check the acc for relational KG
        '''
        first_labels = self.cluster.get_cluster_index(selected_train_E).detach().cpu().numpy()
        second_labels = self.cluster.get_cluster_index(selected_train_L).detach().cpu().numpy()
        all_predicted_labels = list(first_labels) + list(second_labels)
        if data_type == '': # train
            acc_dkm = accuracy_score(all_predicted_labels, self.all_entity_labels_index)
            self.plot_cluster_label_confusion(self.all_entity_labels_index, all_predicted_labels,data_type=data_type)
            print('Differentiable KM accuracy:', acc_dkm)
            first_labels = self.all_entity_labels[:len(selected_train_E)]
            second_labels = self.all_entity_labels[len(selected_train_E):]
        elif data_type == 'test':
            acc_dkm = accuracy_score(all_predicted_labels, self.all_obj_label_test_index)
            self.plot_cluster_label_confusion(self.all_obj_label_test_index, all_predicted_labels,data_type=data_type)
            print('Differentiable KM accuracy:', acc_dkm)
            first_labels = self.all_obj_label_test[:len(selected_train_E)]
            second_labels = self.all_obj_label_test[len(selected_train_E):]

        all_facts = []
        ini_index = 0
        for i in all_relations:
            # if data_type == 'test':
                # if i != self.target_relation:
                    # continue
            single_item = f'{i}[E{first_labels[ini_index]}@E{second_labels[ini_index]}]#'
            all_facts.append(single_item)
            ini_index += 1
        with open(f"{self.current_dir}/cache/{self.task_name}/{self.target_relation}.nl{data_type}", "w") as f:
            for item in all_facts:
                f.write(item)
                f.write('\n')
            f.close()
        return acc_dkm

    def build_kg_ppi(self, data_testing, train_data_E, train_data_L):
        '''
        Build the knowledge graph under the partial predicate invention setting with partial predicate information 
        '''
        all_first_entity_embeddings = train_data_E
        all_second_entity_embeddings = train_data_L
        all_first_symbol = self.cluster.get_cluster_index(all_first_entity_embeddings).detach().cpu().numpy()
        all_second_symbol = self.cluster.get_cluster_index(all_second_entity_embeddings).detach().cpu().numpy()
        all_fact = []
        index = 0
        for item in data_testing['textR']:
            relation = self.move_special_character(str(item))
            single_fact = f'{relation}[E{all_first_symbol[index]}@E{all_second_symbol[index]}]#'
            all_fact.append(single_fact)
            index += 1
        with open(f"{self.current_dir}/cache/{self.task_name}/{self.target_relation}.nl", "w") as f:
            for item in all_fact:
                f.write(item)
                f.write('\n')
            f.close()
        return 0




    def plot_cluster_label_confusion(self, y_true=None, y_pred=None, data_type=''):

        cm = confusion_matrix(y_true, y_pred)
        # Use Hungarian algorithm to reorder clusters
        row_ind, col_ind = linear_sum_assignment(-cm)
        cm = cm[:, col_ind]
        plt.figure(figsize=(8, 6))
        
        if len(y_true) > 20 :
            # do not plot axis and cells
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        else:    
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Cluster {i}' for i in col_ind],
                    yticklabels=[f'Label {i}' for i in np.unique(y_true)])
        plt.xlabel("Predicted Clusters")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix between Clusters and True Labels")
        # save the pics to the output folder 
        plt.savefig(f"{self.current_dir}/cache/{self.task_name}/cluster_label_{self.target_relation}{data_type}.png")
        plt.clf()
        return 0 
    
    def build_target_validation_data(self, fact_set):
        with open(f'{self.current_dir}/cache/{self.task_name}/{self.move_special_character(self.target_relation)}.nl','w') as f:
            for item in fact_set:
                first_entity = self.move_special_character(str(item['text1']))
                relation = self.move_special_character(str(item['textR']))
                second_entity = self.move_special_character(str(item['text2']))
                fact = f"{relation}{self.left_bracket}{first_entity}{self.split}{second_entity}{self.right_bracket}{self.end}"
                print(fact, file=f)
            f.close()
        return 0 
    
    def move_special_character(self, in_str):
        '''
        design function only keep alphabetic and number characters
        '''
        out_str = ''
        stop_lower  = False
        for char in in_str:
            if char.isalnum() or char in ['_','[',']','@']:
                # use lower case for all characters
                if char == '[':
                    stop_lower = True
                if char == ']':
                    stop_lower = False
                if stop_lower == False:
                    char = char.lower()
                out_str = out_str + char
            else:
                out_str = out_str + '_'
        return out_str
    
    def check_acc_PI(self, test_data, existing_rule_set=None):
        '''
        Check the accuracy of learned programs in propositional logic form 
        '''
        rule_set = self.logic_model.interpret(self.valid_unground_atoms, existing_rule_set=existing_rule_set, scale=False)
        metric_obj = self.logic_model.check_metric(test_data, rule_set, remove_low_precision=self.minimal_precision, update_parameters=True, single_check=True)
        rule_set = metric_obj['rule_set']
        acc = metric_obj['metrics']['acc']
        precision = metric_obj['metrics']['rule_set_precision']
        recall = metric_obj['metrics']['rule_set_recall']
        indicators = {'acc': acc, 'precision': precision, 'recall': recall}
        return  indicators, rule_set
        
    def check_acc(self, existing_rule_set = None, relational_image_test = False):
        '''
        Check the accurcy for first-order logic program
        If there is old rule set, then add them into new rule set
        '''
        unground = []
        for item in self.valid_unground_atoms:
            if item == f'{self.target_relation}[{self.target_variable_arrange}]':
                continue
            unground.append(self.move_special_character(item))
        rule_set = self.logic_model.interpret(unground, existing_rule_set= existing_rule_set, scale=False)
        # save realtions
        # print rules 
        # with open(self.rule_path, "w") as f:
        #     print(rule_set,file=f)
        #     f.close()
        # sort all test facts into a file 
        format_all_relation = []
        for item in self.all_relations:
            format_all_relation.append(self.move_special_character(item))
        # check the accuracy
        KG_checker = CheckMetrics(t_relation=self.move_special_character(self.target_relation), task_name = self.task_name, logic_path=self.rule_path, ruleset = rule_set, t_arity=self.target_arity, data_path = f'{self.current_dir}/cache/{self.task_name}/',all_relation=format_all_relation, relational_image_test = relational_image_test)
        # check the accuracy
        acc = KG_checker.check_correctness_of_logic_program(left_bracket=self.left_bracket, right_bracket=self.right_bracket, split=self.split, end_symbol=self.end, split_atom='@', left_atom='[', right_atom=']', minimal_precision=self.minimal_precision)
        return acc, rule_set
    
    def only_check_precision_recall_rules(self, relational_image_test=False):
        with open(self.rule_path+'.pk', "rb") as f:
            rule_set = pickle.load(f)
            f.close()
        acc, rule = self.check_acc(existing_rule_set=rule_set, relational_image_test=relational_image_test)
        # print(rule)
        print("TEST ACC")
        test_recall = acc['recall']
        test_precision = acc['precision']
        print("TEST RECALL", test_recall)
        print("TEST PRECISION", test_precision)
        return 0 

    def save(self, model, optimizer, append_path=''):
        # save the model 
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, self.output_model_path_basic+append_path+'.pt')
        # check if model number is larger then 4, then delete the oldest file
        # model_files = os.listdir(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}")
        # if len(model_files) > 4:
        #     files_with_times = [(self.get_creation_time(f'{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/{x}'), self.get_loss_str(x), x) for x in model_files]
        #     # find the larger loss 
        #     files_with_times.sort(key=lambda x: x[0])
        #     files_with_times.sort(key=lambda x: x[1], reverse=True)
        #     os.remove(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/{files_with_times[0][2]}")


class SubData(Dataset):
    '''
    Process the substitution and raise the random substations 
    '''
    def __init__(self, entity, number_variable, data_length = None):
        self.entity = entity
        self.number_entity = self.entity.shape[0]
        self.number_variable = number_variable
        self.data_length = data_length 
    def __len__(self):
        all_sub_numbers = self.number_entity ** self.number_variable
        min_number = min(self.data_length, all_sub_numbers)
        return min_number if self.data_length != None else all_sub_numbers
    
    def __getitem__(self, idx):
        # get three random indexes 
        index = torch.randint(0, len(self.entity), (self.number_variable,),)
        first = index[0].item()
        second = index[1].item()
        third = index[2].item()
        index_inputs_attention = {'first_entity_id': self.entity[first]["input_ids"],
                                    'second_entity_id': self.entity[second]['input_ids'],
                                    'third_entity_id': self.entity[third]['input_ids'],
                                    'first_entity_mask': self.entity[first]['attention_mask'],
                                    'second_entity_mask': self.entity[second]['attention_mask'],
                                    'third_entity_mask': self.entity[third]['attention_mask']}
        return index_inputs_attention, index






def soft_pro_train_rules(args):
# if __name__ == "__main__":
    current_dir = args.folder_name
    task_name = args.task
    entity_path = f"{current_dir}/cache/{task_name}/all_entities_sampled.csv" # all entities from train file
    relation_path = f"{current_dir}/cache/{task_name}/all_relations_sampled.csv" # all relations from train file
    #initial the model
    model = DifferentiablePropositionalization(entity_path, relation_path, epoch=args.epoch, batch_size = args.batch_size, current_dir=args.folder_name, device=args.device,task_name = args.task, threshold=5000, train_loop_logic=args.train_loop_logic, model_name_specific=args.model_path, ele_number=args.element_number, tokenize_length=args.tokenize_length, load_pretrain_head=args.load_pretrain_head, lr_predicate_model=args.lr_predicate, lr_rule_model=args.lr_rule, target_predicate=args.target_predicate, data_format=args.data_format, cluster_numbers=args.cluster_numbers, lr_dkm = args.lr_dkm, alpha = args.alpha, early_stop=args.early_stop, number_variable=args.number_variable, target_variable_arrange=args.target_variable_arrange,stop_recall=args.stop_recall, substitution_method = args.substitution_method, random_negative = args.random_negative, open_neural_predicate=args.open_neural_predicate, output_file_name=args.output_file_name, minimal_precision=args.minimal_precision, body_arity=args.body_arity, lambda_dkm=args.lambda_dkm, pre_train_ae_num_epochs = args.pre_train_ae_num_epochs)

    # model.soft_pro()
    # model.soft_pro_full_end()
    
    if args.data_format == 'pi':
        model.predicate_invention(128)
    elif args.data_format == 'pi_ae':
        model.predicate_invention_autoencoder(128)
    elif args.data_format == 'ppi':
        model.partial_PI(128)
    elif args.data_format == 'image_ae':
        model.make_tokeniz()
        model.rule_body_train_autoencoder()
        # model.only_check_precision_recall_rules(relational_image_test=True)
    else:
        model.make_tokeniz()
        model.rule_body_train()
        if args.data_format == 'image':
            model.only_check_precision_recall_rules(relational_image_test=True)
        else:
            model.only_check_precision_recall_rules()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help='number of epochs to train neural predicates')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--early_stop", type=int, default=100, help='number of tolerance to train neural predicates')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--folder-name", type=str, default=f'{os.getcwd()}/gammaILP/')
    parser.add_argument('--model-path-parent', type=str, default=f'{os.getcwd()}/gammaILP/cache/icews14/out/')
    parser.add_argument('--model-path', type=str, default='head_only_test_demo_l1', help='specific model path to load the model')
    parser.add_argument('--tokenize_length', type=int, default=10)
    # parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_predicate', type=float, default=1e-4)
    parser.add_argument('--lr_rule', type=float, default=0.5, help='learning rate for the rule model')
    parser.add_argument('--lr_dkm', type=float, default=0.1)
    parser.add_argument('--inf', action='store_true')
    parser.add_argument('--train_loop_logic', type=int, default=1000, help='number of train loop for logic module')
    parser.add_argument('--load_pretrain_head', action='store_true')
    parser.add_argument('--lambda_dkm', type=float, default=1, help='the weight for the dkm loss')
    parser.add_argument('--body_arity', type=str, default='12', help='the body arity for the rule set. 12 indicate consider both unary and binary predicates, 2 indicate only consider binary predicates, 1 indicate only consider unary predicates')
    parser.add_argument('--pre_train_ae_num_epochs', type=int, default=50, help='number of epochs to pre-train the autoencoder')

    # # ! config for sequence images 
    # parser.add_argument('--target_predicate', type=str, default='target', help='target predicate, can be lessthan or predicate inside the dataset')
    # parser.add_argument("--task", type=str, default='mnist_sequence', help='task name. Choose from mnist, relational_images, even, lessthan, kandinsky_onered_all, etc')
    # parser.add_argument('--data_format', type=str, default='image', help='Can be chosen from kg or image or "pi" (predicate invention)')
    # parser.add_argument('--cluster_numbers', type=int, default=10, help='The number of the clusterings centroid')
    # parser.add_argument('--element_number', type=int, default= -1, help='number of elements to sample for training, if the number is less than 0, then the number of elements is the number of all elements')
    # parser.add_argument('--alpha', type=int, default=20, help='the alpha value for the differentiable k-means') 
    # parser.add_argument('--target_variable_arrange', type=str, default='X@X', help='the target variable arrange for the target predicate')
    # parser.add_argument('--number_variable', type=int, default=3, help='the number of variables in the logic program')
    # parser.add_argument('--stop_recall', type=float, default=1.1, help='the target relation for the task')
    # parser.add_argument('--minimal_precision', type=float, default=0.5, help='the least precision the rules in the learned ruleset')
    # parser.add_argument('--substitution_method', type=str, default='chain_random', help='the method to generate the substitution for the body predicates, choose from random, all, chain_random, only chain_random when variable number is 3')
    # parser.add_argument('--random_negative', type=int, default=4, help='Number of batch z when using random substitution method')
    # parser.add_argument('--open_neural_predicate', action='store_true', help='open the neural predicate')
    # parser.add_argument('--output_file_name', type=str, default='chain_substitution_no_neural_predicate', help='the output file name for the rule set.')


    ## ! config for ILP images 
    # parser.add_argument('--target_predicate', type=str, default='pre', help='target predicate, can be lessthan or predicate inside the dataset')
    # parser.add_argument("--task", type=str, default='pre_m', help='task name. Choose from mnist, relational_images, even, even_m, lessthan, kandinsky_onered_all, etc')
    # parser.add_argument('--data_format', type=str, default='image_ae', help='Can be chosen from kg or image or "pi" (predicate invention)')
    # parser.add_argument('--cluster_numbers', type=int, default=10, help='The number of the clusterings centroid')
    # parser.add_argument('--element_number', type=int, default= -1, help='number of elements to sample for training, if the number is less than 0, then the number of elements is the number of all elements')
    # parser.add_argument('--alpha', type=int, default=20, help='the alpha value for the differentiable k-means') 
    # parser.add_argument('--target_variable_arrange', type=str, default='X@Y', help='the target variable arrange for the target predicate')
    # parser.add_argument('--number_variable', type=int, default=3, help='the number of variables in the logic program')
    # parser.add_argument('--stop_recall', type=float, default=1.1, help='the target relation for the task')
    # parser.add_argument('--minimal_precision', type=float, default=0.6, help='the least precision the rules in the learned ruleset')
    # parser.add_argument('--substitution_method', type=str, default='chain_random', help='the method to generate the substitution for the body predicates, choose from random, all, chain_random')
    # parser.add_argument('--random_negative', type=int, default=4, help='Number of batch z when using random substitution method')
    # parser.add_argument('--open_neural_predicate', action='store_true', help='open the neural predicate')
    # parser.add_argument('--output_file_name', type=str, default='chain_substitution_no_neural_predicate', help='the output file name for the rule set.')

    # # ! config for ILP  
    # parser.add_argument('--target_predicate', type=str, default='lessthan', help='target predicate, can be lessthan or predicate inside the dataset')
    # parser.add_argument("--task", type=str, default='lessthan', help='task name. Choose from mnist, relational_images, even, lessthan, kandinsky_onered_all, etc')
    # parser.add_argument('--data_format', type=str, default='kg', help='Can be chosen from kg or image or "pi" (predicate invention)')
    # parser.add_argument('--cluster_numbers', type=int, default=10, help='The number of the clusterings centroid')
    # parser.add_argument('--element_number', type=int, default= -1, help='number of elements to sample for training, if the number is less than 0, then the number of elements is the number of all elements')
    # parser.add_argument('--alpha', type=int, default=20, help='the alpha value for the differentiable k-means') 
    # parser.add_argument('--target_variable_arrange', type=str, default='X@Y', help='the target variable arrange for the target predicate')
    # parser.add_argument('--number_variable', type=int, default=3, help='the number of variables in the logic program')
    # parser.add_argument('--stop_recall', type=float, default=0.9, help='the target relation for the task')
    # parser.add_argument('--minimal_precision', type=float, default=0.5, help='the least precision the rules in the learned ruleset')
    # parser.add_argument('--substitution_method', type=str, default='chain_random', help='the method to generate the substitution for the body predicates, choose from random, all, chain_random')
    # parser.add_argument('--random_negative', type=int, default=4, help='Number of batch z when using random substitution method')
    # parser.add_argument('--open_neural_predicate', action='store_true', help='open the neural predicate')
    # parser.add_argument('--output_file_name', type=str, default='chain_substitution_no_neural_predicate', help='the output file name for the rule set.')
    
    # # # ! config for Kandisky images 
    parser.add_argument('--target_predicate', type=str, default='target', help='target predicate, can be lessthan or predicate inside the dataset')
    parser.add_argument("--task", type=str, default='kandinsky_onetriangle', help='task name. Choose from mnist, relational_images, even, lessthan, kandinsky_onered_all,kandinsky_twopairs_50, , etc')
    parser.add_argument('--data_format', type=str, default='pi_ae', help='Can be chosen from kg or image or pi, pi_ae, image_ae')
    parser.add_argument('--cluster_numbers', type=int, default=9, help='The number of the clusterings centroid')
    parser.add_argument('--element_number', type=int, default= -1, help='number of elements to sample for training, if the number is less than 0, then the number of elements is the number of all elements')
    parser.add_argument('--alpha', type=int, default=20, help='the alpha value for the differentiable k-means') 
    parser.add_argument('--target_variable_arrange', type=str, default='X@X', help='the target variable arrange for the target predicate')
    parser.add_argument('--number_variable', type=int, default=3, help='the number of variables in the logic program')
    parser.add_argument('--stop_recall', type=float, default=1.1, help='the target relation for the task')
    parser.add_argument('--minimal_precision', type=float, default=0.5, help='the least precision the rules in the learned ruleset')
    parser.add_argument('--substitution_method', type=str, default='chain_random', help='the method to generate the substitution for the body predicates, choose from random, all, chain_random')
    parser.add_argument('--random_negative', type=int, default=4, help='Number of batch z when using random substitution method')
    parser.add_argument('--open_neural_predicate', action='store_true', help='open the neural predicate')
    parser.add_argument('--output_file_name', type=str, default='chain_substitution_no_neural_predicate', help='the output file name for the rule set.')
    
    
    # ! config for partial predicate invention task 
    # parser.add_argument('--target_predicate', type=str, default='even', help='target predicate, can be ''lessthan'' in ILP task or ''target'' in the predicate invention task or sequence images task')
    # parser.add_argument("--task", type=str, default='even_ppi', help='task name (the fodler name under the cache/). Choose from mnist, relational_images, even, lessthan, kandinsky_onered_all,half_predicate_half_pi,ppi_sequence_odd, even_ppi, etc')
    # parser.add_argument('--data_format', type=str, default='ppi', help='Can be chosen from [kg, image, pi, ppi], where kg indicate leanring from ILP, image indicate learning from relational image (relational mnist, sequencial minist), pi indicates learning with predicate invention (kandisky), and ppi indicate learning with partial predicate invention task (sequenal mnist)')
    # parser.add_argument('--target_variable_arrange', type=str, default='X@X', help='the target variable arrange for the target predicate')
    # parser.add_argument('--cluster_numbers', type=int, default=10, help='The number of the clusterings centroid, better to promise each similiar entity has at least one cluster')
    # parser.add_argument('--element_number', type=int, default= -1, help='number of elements to sample for training, if the number is less than 0, then the number of elements is the number of all elements')
    # parser.add_argument('--alpha', type=int, default=20, help='the alpha value for the differentiable k-means') 
    # parser.add_argument('--number_variable', type=int, default=3, help='the number of variables in the logic program')
    # parser.add_argument('--stop_recall', type=float, default=1.1, help='the target relation for the task')
    # parser.add_argument('--minimal_precision', type=float, default=0.2, help='the least precision the rules in the learned ruleset')
    # parser.add_argument('--substitution_method', type=str, default='chain_random', help='the method to generate the substitution for the body predicates, choose from random, all, chain_random')
    # parser.add_argument('--random_negative', type=int, default=4, help='Number of batch z when using random substitution method')
    # parser.add_argument('--open_neural_predicate', action='store_true', help='open the neural predicate')
    # parser.add_argument('--output_file_name', type=str, default='chain_substitution_no_neural_predicate', help='the output file name for the rule set.')
    start_time = time.time()
    args = parser.parse_args()
    print(args)
    # torch.manual_seed(100)
    print("CUDA seed:", torch.cuda.initial_seed())
    # pre_train_predicate_neural(args)
    soft_pro_train_rules(args)
    end_time = time.time()
    running_time = end_time - start_time
    # conver running time to second 
    print(f"Running time: {running_time:.2f} seconds")



# todo The problem may appear at two stage:
# todo 1. The distance predicate is not accurate 
# todo 2. The number of substitution is not enough
# todo 3. train the distance predicate and logic program together 
#todo 4. Using the cluster to get the semantics and build the semantics graph to check the accuracy of rules. ✔️
# todo 5. Show the clustering accuracy and debug the accuracy 
# todo 6. for the succ dataset, the random variable is less to make succ holds. Desgin an algorithm to make the predicate hold and unhold at 50% percent. 
#todo exp: all with no neural predicate. all with neural predicate and check the accuracy of neural predicate. all on KG. all on image KG. all on realistic image dataset. 
