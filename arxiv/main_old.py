import sys
import os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)
import pickle
from ordered_set import OrderedSet
import datetime
import torch 
import itertools
import torch.nn as nn
import argparse
from torch.utils.data import  Dataset
from torch.optim import AdamW
import datasets
from datasets import load_dataset, load_from_disk
from transformers import  BertTokenizer, BertForMaskedLM
from torch.utils.tensorboard import SummaryWriter
import tqdm
from sklearn.metrics import confusion_matrix
from logic_back.dforl_torch import DeepRuleLayer
from logic_back.metrics_checker import CheckMetrics
from code.compgraph import DkmCompGraph
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
'''
This version implement the bert embeddings and feedforward neural network for the propositionalization layer
'''

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
    def __init__(self, entity_path, relation_path, label_path, element_number = 10, random_negative = 20, target_atoms_index = [], target_predicate_arity = 2):
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
        
        
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, idx):
        # get the knowledge base into GPU 
        idx_mod = idx % len(self.selected_target_atom_index)
        idx_item = self.selected_target_atom_index[idx_mod]
    
        random_z_idx = torch.randint(0,self.all_objects_length, (self.random_negative,))
        idx_batch = torch.ones_like(random_z_idx) * idx_item
        idx_batch = idx_batch.unsqueeze(1)
        random_z_idx = random_z_idx.unsqueeze(1)
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
        another_random_z_idx = torch.randint(0, self.all_objects_length, (self.random_negative,))
        another_random_z_idx_1 = torch.randint(0, self.all_objects_length, (self.random_negative,))
        another_random_z_idx_2 = torch.randint(0, self.all_objects_length, (self.random_negative,))
        # combine Z substitutions 
        random_z_idx = torch.cat([random_z_idx, another_random_z_idx.unsqueeze(1), another_random_z_idx_1.unsqueeze(1), another_random_z_idx_2.unsqueeze(1)], dim=0)
        # combine X, Y, Z substitutions together 
        if self.target_predicate_arity == 1:
            idx_batch_connect = torch.cat([idx_batch,negative_first_entity_index.unsqueeze(1)], dim=0)
            second_entity_index = torch.cat([another_random_z_idx.unsqueeze(1), another_random_z_idx.unsqueeze(1)], dim=0)
            random_z_idx = torch.cat([another_random_z_idx_1.unsqueeze(1), another_random_z_idx_2.unsqueeze(1)], dim=0)
        index = torch.cat([idx_batch_connect, second_entity_index, random_z_idx], dim=1)
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
        index_to_obj = {}
        for key, value in index_to_entity.items():
            index_to_obj[key] = value
        self.index_to_obj = index_to_obj
        if self.target_predicate_arity == 1:
            self.target_entity = [self.index_to_obj[i] for i in self.selected_target_atom_index]
            self.target_entity_index = [entity_to_index[i][0] for i in self.target_entity]
            self.negative_entity = [i for i in self.entity_to_index.keys() if i not in self.target_entity]
            self.negative_entity_index = [entity_to_index[i][0] for i in self.negative_entity]
            
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, idx):
        # get the knowledge base into GPU 
        idx_mod = idx % len(self.selected_target_atom_index)
        idx_item = self.selected_target_atom_index[idx_mod]
        idx_batch = torch.ones(self.random_negative, dtype=torch.int64) * idx_item
        idx_batch = idx_batch.unsqueeze(1)

        
        if self.target_predicate_arity == 2:
            first_entities_num = len(self.entity)
            second_entity_index = first_entities_num + idx_batch
            # get the connected index by first entity and second entity 
            fist_entity = self.index_to_entity[idx_batch[0][0].item()]
            seocnd_entity = self.index_to_entity[second_entity_index[0][0].item()]
            connected_index_XY = torch.tensor(self.entity_to_entityindex[fist_entity] + self.entity_to_entityindex[seocnd_entity])
            random_Z_XY_index = torch.randint(0, connected_index_XY.shape[0], (self.random_negative,))
            random_Z_XY = connected_index_XY[random_Z_XY_index].unsqueeze(1)
            random_Z_XY_index_1 = torch.randint(0, connected_index_XY.shape[0], (self.random_negative,))
            random_Z_XY_1 = connected_index_XY[random_Z_XY_index_1].unsqueeze(1)
            
            # get random Z from all objects
            random_z_idx = torch.randint(0,self.all_objects_length, (self.random_negative,)).unsqueeze(1)
            
            
            
            
            # find the random object to substitute the variable X 
            negative_first_entity_index = torch.randint(0, self.all_objects_length, (self.random_negative,))
            # no fact in the random X
            negative_first_entity_index = torch.where(negative_first_entity_index == idx_batch[0][0], (negative_first_entity_index + 1) % self.all_objects_length, negative_first_entity_index).unsqueeze(1)
            
            # find the random object to substitute the variable Y
            negative_second_entity_index = torch.randint(0, self.all_objects_length, (self.random_negative,))
            # no fact in the random Y
            negative_second_entity_index = torch.where(negative_second_entity_index == second_entity_index[0][0], (negative_second_entity_index + 1) % self.all_objects_length, negative_second_entity_index).unsqueeze(1)
            
            # connect X substitutions (correct X Y R_conXY(Z), correct X Y R_conXY(Z), R_disconY(X)Y R_con(XYZ), XR_disconX(Y)R_conXY(Z))
            idx_batch_connect  = torch.cat([idx_batch, idx_batch, negative_first_entity_index, idx_batch], dim=0)
            
            # connect Y substitutions 
            second_entity_index = torch.cat([second_entity_index,second_entity_index,second_entity_index, negative_second_entity_index], dim=0)

            # get the random objects to substitute the variable Z
            another_random_z_idx_1 = torch.randint(0, self.all_objects_length, (self.random_negative,)).unsqueeze(1)
            another_random_z_idx_2 = torch.randint(0, self.all_objects_length, (self.random_negative,)).unsqueeze(1)
            
            random_z_idx = torch.cat([random_Z_XY, random_Z_XY_1, another_random_z_idx_1, another_random_z_idx_2], dim=0)
            
            # combine X, Y, Z substitutions together
            index = torch.cat([idx_batch_connect, second_entity_index, random_z_idx], dim=1)
        elif self.target_predicate_arity == 1:
            # todo: The substitutation are generated based on the chain law
            # connect X substitutions (correct X R_conX(Y) R_conXY(Z),  R(X)R(Y)R(Z))
            # get the connected index by first entity and second entity 
            fist_entity = self.index_to_entity[idx_batch[0][0].item()]
            connected_index_X = self.entity_to_entityindex[fist_entity]
            Y_obj = list(set([self.index_to_obj[i] for i in connected_index_X]))
            connected_index_X = torch.tensor([self.entity_to_index[i][0] for i in Y_obj])
            
            random_Y_cX_index = torch.randint(0, connected_index_X.shape[0], (self.random_negative,))
            random_Y_cx = connected_index_X[random_Y_cX_index].unsqueeze(1)
            random_Y_cx_list = random_Y_cx.reshape(-1).numpy().tolist()
                
            # get all index Y 
            possible_Y = list(set([self.index_to_obj[i] for i in random_Y_cx_list]))
            # possible_Y = self.index_to_obj[random_Y_cx].reshape(-1).numpy().tolist()
            
            connected_z_tensors = [self.entity_to_entityindex[k] for k in possible_Y if k in self.entity_to_entityindex]
            possible_Z_pool = []
            for tensor in connected_z_tensors:
                possible_Z_pool.extend(tensor)
            # possible z obj  
            z_obj  = list(set([self.index_to_obj[i] for i in possible_Z_pool]))
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
            neg_Y_obj = list(set([self.index_to_obj[i] for i in neg_connected_index_X]))
            neg_connected_index_X = torch.tensor([self.entity_to_index[i][0] for i in neg_Y_obj])
            
            neg_random_Y_cX_index = torch.randint(0, neg_connected_index_X.size(0), (self.random_negative,))
            neg_random_Y_cx = neg_connected_index_X[neg_random_Y_cX_index].unsqueeze(1)
            neg_random_Y_cx_list = neg_random_Y_cx.reshape(-1).numpy().tolist()
            
            
            # get all index Y 
            neg_possible_Y = list(set([self.index_to_obj[i] for i in neg_random_Y_cx_list]))            
            neg_connected_z_tensors = [self.entity_to_entityindex[k] for k in neg_possible_Y if k in self.entity_to_entityindex]
            neg_possible_Z_pool = []
            for tensor in neg_connected_z_tensors:
                neg_possible_Z_pool.extend(tensor)
            # possible z obj  
            neg_z_obj  = list(set([self.index_to_obj[i] for i in neg_possible_Z_pool]))
            neg_possible_Z_pool = np.array([self.entity_to_index[i][0] for i in neg_z_obj])
            neg_random_Z = torch.randint(0, len(neg_possible_Z_pool), (self.random_negative,))
            neg_possible_Z_pool = torch.tensor(neg_possible_Z_pool)
            neg_random_z_index = neg_possible_Z_pool[neg_random_Z].unsqueeze(1)
            
            
            
            idx_batch_connect = torch.cat([idx_batch,negative_index], dim=0)
            second_entity_index = torch.cat([random_Y_cx, neg_random_Y_cx], dim=0)
            random_z_idx = torch.cat([random_z_index, neg_random_z_index], dim=0)
            index = torch.cat([idx_batch_connect, second_entity_index, random_z_idx], dim=1)
        return index
    

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



class DifferentiablePropositionalization(nn.Module, BaseRule):
    def __init__(self, entity_path, relation_path, epoch=10, batch_size =512, early_stop = 5, ele_number=10, current_dir='',device='', task_name='',final_dim = 10,threshold = 1500, train_loop_logic=10, model_name_specific = None, tokenize_length=10, load_pretrain_head=True, lr_predicate_model = 1e-4, lr_rule_model = 1e-3, lr_dkm  = 1e-4, target_predicate = 'Host a visit', data_format = 'kg', cluster_numbers = 10, alpha=1, number_variable=3, target_variable_arrange = 'X@Y', stop_recall = 0.97, substitution_method = 'random', random_negative = 32, open_neural_predicate = False, output_file_name = '', minimal_precision = 0.5):
        super(DifferentiablePropositionalization, self).__init__()
        BaseRule.__init__(self, ele_number, final_dim, current_dir, task_name,tokenize_length)
        # import bert and token 
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
        self.mapping = {0:'X', 1:'Y', 2:'Z'}
        self.left_bracket = '['
        self.right_bracket = ']'
        self.split = '@'
        self.end = '#'
        
        ## TODO user defined based on the task
        self.number_variable = number_variable
        self.variable_perturbations_list = list(itertools.permutations(range(self.number_variable), 2)) + [(0,0), (1,1), (2,2)]
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
        
        if self.data_format == 'image':
            self.n_clusters = cluster_numbers
            self.alpha = alpha

    def make_tokeniz(self):
        '''
        # ! Run this function in the first time 
        '''
        # read train data and generated entity and relation

        train_data = load_dataset('csv', data_files = f"{self.current_dir}/cache/{self.task_name}/embeddings.csv")['train']

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
        
    
    def rule_body_train(self):
        '''
        Read all entities and process and propositionalalization and rule learning task together.
        Train distance and logic model together.
        BERT embedding is fix but head embeddings is trainable.
        '''
        # read all entities 
        if self.data_format == 'kg':
            train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/embeddings.csv")['train']
        elif self.data_format == 'image':
            train_data = load_dataset('csv',  data_files=f"{self.current_dir}/cache/{self.task_name}/embeddings.csv")['train']
            # get all text 1
            if self.ele_number > 0:
                all_label1 = [i for i in train_data['text1'][:self.ele_number]]
                all_label2 = [ i for i in train_data['text2'][:self.ele_number]]
            else:
                all_label1 = [i for i in train_data['text1']]
                all_label2 = [ i for i in train_data['text2']]
            self.all_entity_labels = all_label1 + all_label2
            all_relations_in_order = [i for i in train_data['textR']]
            self.all_relation_index = torch.tensor([i for i in range(len(all_relations_in_order))])
            
        else:
            raise('The data format is wrong')
        relation_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/relation_token.hf")
        target_atoms_index = []
        ini_index = 0
        for item in train_data:
            if item['textR'] == self.target_relation:
                target_atoms_index.append(ini_index)
            ini_index = ini_index + 1
        self.tar_predicate_index = None
        number_relations = len(relation_data)
        self.all_relations = relation_data['text']
        print('All relations:', self.all_relations) 

        # get predicate arity 
        predicate_arity = self.check_arity_predicate(train_data)
        print('Predicate arity:', predicate_arity)

        # get the entity to entity_index list 
        self.entity_entityindex, self.index_to_entity, self.entity_index = self.get_connected_entity(train_data)
        
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
        
        # find the target atom index in the valid unground atoms
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
        if self.substitution_method == 'random':
            tuple_data = BatchZ(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity)
            substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True)
        elif self.substitution_method == 'chain_random':
            tuple_data = BatchChain(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index, target_predicate_arity = self.target_predicate_arity,index_to_entity=self.index_to_entity, entity_to_entityindex=self.entity_entityindex, entity_to_index = self.entity_index)
            substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=True)
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
            print(kmeans_model.predict(all_obj_bert_embeddings_numpy))
            print('[Labels]')
            print(self.all_entity_labels)
            if type(self.all_entity_labels[0]) == str:
                # Create a mapping: string → unique int
                str_to_int = {s: i for i, s in enumerate(sorted(set(self.all_entity_labels)))}
                # Map each string to its corresponding int
                int_list = [str_to_int[s] for s in self.all_entity_labels]
                print(int_list)  # e.g., [0, 1, 0, 2, 1]
                self.all_entity_labels = int_list
            acc_initial = accuracy_score(kmeans_model.labels_, self.all_entity_labels)
            self.plot_cluster_label_confusion(y_true=self.all_entity_labels,y_pred=kmeans_model.labels_)
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


        # start training
        self.logic_model.train()
        head_model.train()
        if self.data_format == 'image':
            self.cluster.train()
        rule_set = None
        for single_epoch in range(self.epoch):
            total_loss_rule = 0
            total_loss = 0
            total_dkm_loss = 0
            total_body_loss = 0
            for index in tqdm.tqdm(substation_data):
                if self.substitution_method == 'all':
                    index = torch.stack(index)
                    index = index.transpose(0,1)
                logic_optimizer.zero_grad()
                index = index.to(self.device)
                index = index.squeeze(0)
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
                
                
                
                
                
                # first find labels for all perturbation 
                #! Based on the all body predicates and entity pairs, generate the labels 
                perturbations_bert_embeddings = all_obj_bert_embeddings[perturbations]
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
                labels_body = (triples==all_facts_bert_embeddings_var).all(dim=2).any(dim=1)
                labels_body = labels_body.reshape(number_instance, -1).float()
                
                
                # get the head embeddings for all perturbations 
                # todo the number of false head is missing 
                # todo all body are all zero because the limited z 
                
                # get the head embedding of the entity 
                if self.data_format == 'image':
                    #! for image task, we need to use the cluster centeriod as the entity embedding 
                    selected_train_E_head_embeddings, l1 = self.cluster(selected_train_E)
                    selected_train_L_head_embeddings, l2 = self.cluster(selected_train_L)
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
                
                perturbations_embeddings = all_objects_head_embedding[perturbations]
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
                ground_truth_values = ground_truth_values[:,self.valid_atom_index]
                labels_body = labels_body[:,self.valid_atom_index]
                body_loss = torch.nn.BCELoss()(ground_truth_values, labels_body) # can use Cross Entropy Loss
                
                #! the threshold is human-defined here now 
                # ! Do not binaries the values for the rule learning model 
                # threshold = 0.5
                # ground_truth_values = torch.where(ground_truth_values > threshold , torch.ones_like(ground_truth_values), torch.zeros_like(ground_truth_values))
                if self.open_neural_predicate == True:
                    pass
                else:
                    ground_truth_values = labels_body
                head_predicate_labels_predicated = ground_truth_values[:,self.target_atom_index].unsqueeze(1)
                body_predicate_labels_predicated = torch.cat([ground_truth_values[:,:self.target_atom_index], ground_truth_values[:,self.target_atom_index+1:]],dim=1)

                # using logic model and body atoms to predict the head predicate
                predicated = self.logic_model(body_predicate_labels_predicated)
                loss_rule = torch.nn.MSELoss()(predicated, head_predicate_labels_predicated)

                #loss for dkm 
                if self.data_format == 'image':
                    dkm_loss = l1 + l2
                else: 
                    dkm_loss = 0
                
                # all loss from logic rule loss, body predicate, and [deep k-means]
                if self.open_neural_predicate == True:
                    loss = 0.5*loss_rule + 0.5*body_loss + 0.5 * dkm_loss
                else:
                    loss = 0.5*loss_rule + 0.5 * dkm_loss
                

                loss.backward(retain_graph=True)
                # todo define the loss from neural predicate here 
                logic_optimizer.step()
                total_loss_rule = total_loss_rule + loss_rule.item()
                total_loss += loss.item()
                total_body_loss += body_loss.item()
                if self.data_format == 'image':
                    total_dkm_loss += dkm_loss.item()
                else:
                    total_dkm_loss += 0

            # if the input is the image, build the knowledge graphs based on the clustering centroid as the entities and the existing relations as the predicates
            dkm_acc = 0
            if self.data_format == 'image':
                dkm_acc = self.build_kg_based_on_cluster(selected_train_E, selected_train_L, all_relations_in_order)
                

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
            acc, rule_set = self.check_acc(rule_set)
            # save rule set 
            with open(self.rule_path+'.pk', "wb") as f:
                pickle.dump(rule_set, f)
                f.close()
            precision = acc['precision']
            recall = acc['recall']
            if recall >= self.stop_recall:
                break
        self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
        print("[Training Finished]")
        print("[Rules]")
        print(rule_set)
        print(f"[Precision]")
        print(precision)
        print(f"[Recall]")
        print(recall)
        return 0 

    def build_kg_based_on_cluster(self, selected_train_E, selected_train_L, all_relations):
        first_labels = self.cluster.get_cluster_index(selected_train_E)
        second_labels = self.cluster.get_cluster_index(selected_train_L)
        all_predicted_labels = list(first_labels) + list(second_labels)
        acc_dkm = accuracy_score(all_predicted_labels, self.all_entity_labels)
        self.plot_cluster_label_confusion(self.all_entity_labels, all_predicted_labels)
        print('Differentiable KM accuracy:', acc_dkm)
        all_facts = []
        ini_index = 0
        for i in all_relations:
            single_item = f'{i}[E{first_labels[ini_index]}@E{second_labels[ini_index]}]#'
            all_facts.append(single_item)
            ini_index += 1
        with open(f"{self.current_dir}/cache/{self.task_name}/{self.target_relation}.nl", "w") as f:
            for item in all_facts:
                f.write(item)
                f.write('\n')
            f.close()
        return acc_dkm

    def plot_cluster_label_confusion(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        # Use Hungarian algorithm to reorder clusters
        row_ind, col_ind = linear_sum_assignment(-cm)
        cm = cm[:, col_ind]
        plt.figure(figsize=(8, 6))
        
        if len(y_true) > 20:
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
        plt.savefig(f"{self.current_dir}/cache/{self.task_name}/cluster_label_{self.target_relation}.png")
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
        
        
    def check_acc(self, existing_rule_set = None):
        '''
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
        KG_checker = CheckMetrics(t_relation=self.move_special_character(self.target_relation), task_name = self.task_name, logic_path=self.rule_path, ruleset = rule_set, t_arity=self.target_arity, data_path = f'{self.current_dir}/cache/{self.task_name}/',all_relation=format_all_relation)
        # check the accuracy
        acc = KG_checker.check_correctness_of_logic_program(left_bracket=self.left_bracket, right_bracket=self.right_bracket, split=self.split, end_symbol=self.end, split_atom='@', left_atom='[', right_atom=']', minimal_precision=self.minimal_precision)  
        return acc, rule_set
    
    def only_check_precision_recall_rules(self):
        with open(self.rule_path+'.pk', "rb") as f:
            rule_set = pickle.load(f)
            f.close()
        self.check_acc(existing_rule_set=rule_set)

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
    model = DifferentiablePropositionalization(entity_path, relation_path, epoch=args.epoch, batch_size =2, current_dir=args.folder_name, device=args.device,task_name = args.task, threshold=5000, train_loop_logic=args.train_loop_logic, model_name_specific=args.model_path, ele_number=args.element_number, tokenize_length=args.tokenize_length, load_pretrain_head=args.load_pretrain_head, lr_predicate_model=args.lr_predicate, lr_rule_model=args.lr_rule, target_predicate=args.target_predicate, data_format=args.data_format, cluster_numbers=args.cluster_numbers, lr_dkm = args.lr_dkm, alpha = args.alpha, early_stop=args.early_stop, number_variable=args.number_variable, target_variable_arrange=args.target_variable_arrange,stop_recall=args.stop_recall, substitution_method = args.substitution_method, random_negative = args.random_negative, open_neural_predicate=args.open_neural_predicate, output_file_name=args.output_file_name, minimal_precision=args.minimal_precision)
    model.make_tokeniz()
    # model.soft_pro()
    # model.soft_pro_full_end()
    model.rule_body_train()
    model.only_check_precision_recall_rules()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200, help='number of epochs to train neural predicates')
    parser.add_argument("--batch_size", type=int, default=5096)
    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--early_stop", type=int, default=100, help='number of tolerance to train neural predicates')
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--folder-name", type=str, default=f'{os.getcwd()}/gammaILP/')
    parser.add_argument('--model-path-parent', type=str, default=f'{os.getcwd()}/gammaILP/cache/icews14/out/')
    parser.add_argument('--model-path', type=str, default='head_only_test_demo_l1', help='specific model path to load the model')
    parser.add_argument('--tokenize_length', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_predicate', type=float, default=1e-4)
    parser.add_argument('--lr_rule', type=float, default=0.05)
    parser.add_argument('--lr_dkm', type=float, default=0.1)
    parser.add_argument('--inf', action='store_true')
    parser.add_argument('--train_loop_logic', type=int, default=1000, help='number of train loop for logic module')
    parser.add_argument('--load_pretrain_head', action='store_true')
    
    
    # ! config for images 
    parser.add_argument('--target_predicate', type=str, default='lessthan', help='target predicate, can be lessthan or predicate inside the dataset')
    parser.add_argument("--task", type=str, default='lessthan', help='task name. Choose from mnist, relational_images, even, lessthan....')
    parser.add_argument('--data_format', type=str, default='image', help='Can be chosen from kg or image')
    parser.add_argument('--cluster_numbers', type=int, default=50, help='The number of the clusterings centroid')
    parser.add_argument('--element_number', type=int, default= -1, help='number of elements to sample for training, if the number is less than 0, then the number of elements is the number of all elements')
    parser.add_argument('--alpha', type=int, default=10, help='the alpha value for the differentiable k-means') 

    
    # ! update frequently based on the task 
    # parser.add_argument('--data_format', type=str, default='kg', help='Can be chosen from kg or image')    
    # parser.add_argument('--target_predicate', type=str, default='member', help='target predicate')
    # parser.add_argument("--task", type=str, default='member')
    parser.add_argument('--target_variable_arrange', type=str, default='X@Y', help='the target variable arrange for the target predicate')
    parser.add_argument('--number_variable', type=int, default=3, help='the number of variables in the logic program')
    parser.add_argument('--stop_recall', type=float, default=1, help='the target relation for the task')
    parser.add_argument('--minimal_precision', type=float, default=0.2, help='the least precision the rules in the learned ruleset')
    parser.add_argument('--substitution_method', type=str, default='chain_random', help='the method to generate the substitution for the body predicates, choose from random, all, chain_random')
    parser.add_argument('--random_negative', type=int, default=4, help='Number of batch z when using random substitution method')
    parser.add_argument('--open_neural_predicate', action='store_true', help='open the neural predicate')
    parser.add_argument('--output_file_name', type=str, default='chain_substitution_no_neural_predicate', help='the output file name for the rule set.')
    
    
    
    # parser.add_argument('--element_number', type=int, default= 100, help='number of elements to sample for training')
    args = parser.parse_args()
    # record the parameters 
    print(args)
    # torch.manual_seed(1)
    # pre_train_predicate_neural(args)
    soft_pro_train_rules(args)



# todo The problem may appear at two stage:
# todo 1. The distance predicate is not accurate 
# todo 2. The number of substitution is not enough
# todo 3. train the distance predicate and logic program together 
#todo 4. Using the cluster to get the semantics and build the semantics graph to check the accuracy of rules. ✔️
# todo 5. Show the clustering accuracy and debug the accuracy 
# todo 6. for the succ dataset, the random variable is less to make succ holds. Desgin an algorithm to make the predicate hold and unhold at 50% percent. 
#todo exp: all with no neural predicate. all with neural predicate and check the accuracy of neural predicate. all on KG. all on image KG. all on realistic image dataset. 