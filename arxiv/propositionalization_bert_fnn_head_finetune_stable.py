import sys
import os
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)
import pickle
from ordered_set import OrderedSet
import datetime
import pandas as pd
import torch 
import itertools
import torch.nn as nn
import argparse
from accelerate import Accelerator
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments
import datasets
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel, BertTokenizer, BertForMaskedLM
from torch.utils.tensorboard import SummaryWriter
import tqdm
from sklearn.metrics import confusion_matrix
from logic_back.dforl_torch import DeepRuleLayer
from logic_back.metrics_checker import CheckMetrics
import aix360_k.aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier as trxf_classifier
import time
from code.compgraph import DkmCompGraph
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
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
    def __init__(self, output_dim = 100, second_dim = 10, cluster:DkmCompGraph = None):
        super(FactHead, self).__init__()
        self.linear = nn.Linear(768, output_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, second_dim)
        self.last = nn.Linear(3*second_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.cluster = cluster
        
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
        if type(self.cluster) == None:
            e1 = self.generate_embedding(embed_e1)
            e2 = self.generate_embedding(embed_e2)
            dkm_loss = 0
        else:
            e1,p1,loss_e1 = self.get_center(embed_e1)
            e2,p2,loss_e2 = self.get_center(embed_e2)
            dkm_loss = loss_e1 + loss_e2
                    
        embed_r = self.generate_embedding(embed_r)
        
        x = self.get_possible(e1= e1, embed_r=embed_r, e2=e2)
        return x, dkm_loss

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

class BatchNegative(Dataset):
    '''
    For each positive sample, generate a batch negative for improving the performance 
    Only find the entities in the target relations. Use a random y to replace the target 
    '''
    def __init__(self, entity_path, relation_path, label_path, element_number = 10, random_negative = 20, target_atoms_index = []):
        # read tensor data
        if type(entity_path) == str:
            self.entity = torch.load(entity_path) 
            self.relation = torch.load(relation_path)
            self.label = torch.load(label_path)
        else:
            self.entity = entity_path
            self.relation = relation_path
            self.label = label_path
        self.entity = self.entity[:element_number]
        self.relation = self.relation[:element_number]
        self.label = self.label[:element_number]
        self.all_objects = torch.cat([self.entity, self.label], dim=0)
        self.selected_target_atom_index = []
        for item in target_atoms_index:
            if item < element_number:
                self.selected_target_atom_index.append(item)

        self.all_objects_length = self.all_objects.shape[0]
        self.min_length = min(len(d) for d in [self.entity, self.relation, self.label])
        self.random_negative = random_negative
        # build the fact KB 
        self.all_facts = torch.concat([torch.cat([self.entity, self.relation], dim=1), self.label], dim=1)
        
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, idx):
        # get the knowledge base into GPU 
        idx = idx % len(self.selected_target_atom_index)
        idx = self.selected_target_atom_index[idx]
        
        data_e = self.entity[idx]
        data_r = self.relation[idx]
        data_l = self.label[idx]
        
        random_z_idx = torch.randint(0,self.all_objects_length, (self.random_negative,))
        idx_batch = torch.ones_like(random_z_idx) * idx
        random_position = torch.randint(0,2, (self.random_negative,))

        
        random_index = torch.randint(0, self.min_length, (self.random_negative,))
        
        # replace idx in random index to avoid the same index
        random_index = torch.where(random_index == idx, (random_index + 1) % self.min_length, random_index)
        
        # if random_index == idx:
            # random_index = (random_index + 1) % self.min_length
        composed_head = self.entity[idx_batch]
        composed_tail  = self.label[idx_batch]
        all_random_head = self.entity[random_index]
        all_random_tail = self.label[random_index]
        
        # replace the random index with the composed head and tail
        random_position = random_position.unsqueeze(1)
        random_position = random_position.expand(-1, composed_head.shape[1])
        neg_head = torch.where(random_position == torch.ones([1, composed_head.shape[1]]) , all_random_head, composed_head)
        neg_tail = torch.where(random_position == torch.ones([1, composed_head.shape[1]]), composed_tail, all_random_tail)

        # add a filter to choose real false examples 
        # neg_facts = torch.cat([neg_head, self.relation[idx_batch], neg_tail], dim=1)
        # neg_facts = torch.unique(neg_facts, dim=0)
        # neg_facts_sub = neg_facts.unsqueeze(1)
        # all_fact_all = self.all_facts.unsqueeze(0)
        # existing = (neg_facts_sub==all_fact_all).all(dim=2).any(dim=1)
        # neg_fact_final = neg_facts[~existing]
                
        # all relations caomes, add variables as random z, to check them true of false, train the predicate and rule at same time at this stage , z,x,z,y,x,z, 
        #! can be merger to forward computations 
            
        random_z_obj = self.all_objects[random_z_idx]
        idx_batch = idx_batch.unsqueeze(1)
        random_z_idx = random_z_idx.unsqueeze(1)
        
        first_entities_num = len(self.entity)
        second_entity_index = first_entities_num + idx_batch
        index = torch.cat([idx_batch, second_entity_index, random_z_idx], dim=1)
        return {
            "embed_e": data_e,
            "embed_r": data_r,
            "embed_l": data_l,
            "neg_head_embed": neg_head,
            "neg_tail_embed": neg_tail, 
            "random_z_obj": random_z_obj
        }, index




class BatchZ(Dataset):
    '''
    Random generate random Z objects based on all X, Y in positive examples with target predicates 
    '''
    def __init__(self, entity_path, relation_path, label_path, element_number = 10, random_negative = 20, target_atoms_index = []):
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
        
        
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, idx):
        # get the knowledge base into GPU 
        idx = idx % len(self.selected_target_atom_index)
        idx = self.selected_target_atom_index[idx]
    
        random_z_idx = torch.randint(0,self.all_objects_length, (self.random_negative,))
        idx_batch = torch.ones_like(random_z_idx) * idx
        idx_batch = idx_batch.unsqueeze(1)
        random_z_idx = random_z_idx.unsqueeze(1)
        
        first_entities_num = len(self.entity)
        second_entity_index = first_entities_num + idx_batch
        
        negative_second_entity_index = torch.randint(0, self.all_objects_length, (self.random_negative,))
        negative_second_entity_index = torch.where(negative_second_entity_index == second_entity_index[0][0], (negative_second_entity_index + 1) % self.all_objects_length, negative_second_entity_index)
        idx_batch  = torch.cat([idx_batch, idx_batch], dim=0)
        second_entity_index = torch.cat([second_entity_index, negative_second_entity_index.unsqueeze(1)], dim=0)
        another_random_z_idx = torch.randint(0, self.all_objects_length, (self.random_negative,))
        random_z_idx = torch.cat([random_z_idx, another_random_z_idx.unsqueeze(1)], dim=0)
        
        index = torch.cat([idx_batch, second_entity_index, random_z_idx], dim=1)
        return index



class PNEmbeddings(Dataset):
    def __init__(self, entity_path, relation_path, label_path, element_number = 10):
        # read tessor data
        self.entity = torch.load(entity_path) 
        self.relation = torch.load(relation_path)
        self.label = torch.load(label_path)
        self.entity = self.entity[:element_number]
        self.relation = self.relation[:element_number]
        self.label = self.label[:element_number]
        self.all_objects = torch.cat([self.entity, self.label], dim=0)
        self.all_objects_length = self.all_objects.shape[0]
        self.min_length = min(len(d) for d in [self.entity, self.relation, self.label])
        
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, idx):
        data_e = self.entity[idx]
        data_r = self.relation[idx]
        data_l = self.label[idx]
        
        random_z_idx = torch.randint(0,self.all_objects_length, (1,)).item()
        
        random_position = torch.randint(0,2, (1,)).item()
        random_index = torch.randint(0, self.min_length, (1,)).item()
        
        if random_index == idx:
            random_index = (random_index + 1) % self.min_length
        if random_position == 1:
            neg_head = self.entity[random_index]
            neg_tail = self.label[idx]
        else:
            neg_head = self.entity[idx]
            neg_tail = self.label[random_index]
            
        random_z_obj = self.all_objects[random_z_idx]
        index = torch.tensor([idx, idx, random_z_idx])
        return {
            "embed_e": data_e,
            "embed_r": data_r,
            "embed_l": data_l,
            "neg_head_embed": neg_head,
            "neg_tail_embed": neg_tail, 
            "random_z_obj": random_z_obj
        }, index
        

class ParallelDataset(Dataset):
    '''
    This dataset return random tuple for a relation 
    '''
    def __init__(self, e, r, l):
        self.e = e
        self.l = l 
        self.r = r 
        self.min_length = min(len(d) for d in [e,r,l])  # Use the smallest dataset size

    def __len__(self):
        return self.min_length  # Prevent index errors

    def __getitem__(self, idx):
        data1 = self.e[idx]
        data2 = self.r[idx]
        data3 = self.l[idx]
        
        # return random index from 0 to min_length 
        random_head = torch.randint(0, 2, (1,)).item()
        random_index = torch.randint(0, self.min_length, (1,)).item()
        if random_index == idx:
            random_index = (random_index + 1) % self.min_length
        if random_head == 1:
            neg_head = self.e[random_index]
            neg_tail = self.l[idx]
        else:
            neg_head = self.e[idx]
            neg_tail = self.l[random_index]
        
        assert data1["input_ids_textE"].shape == data2["input_ids_textR"].shape == data3["input_ids_label"].shape == neg_head["input_ids_textE"].shape == neg_tail["input_ids_label"].shape
        assert data1["attention_mask_textE"].shape == data2["attention_mask_textR"].shape == data3["attention_mask_label"].shape == neg_head["attention_mask_textE"].shape == neg_tail["attention_mask_label"].shape

        return {
            "id_e": data1["input_ids_textE"],
            "mask_e": data1["attention_mask_textE"],
            "id_r": data2["input_ids_textR"],
            "mask_r": data2["attention_mask_textR"],
            "id_l": data3["input_ids_label"],
            "mask_l": data3["attention_mask_label"],
            "neg_head_id": neg_head["input_ids_textE"],
            "neg_head_mask": neg_head["attention_mask_textE"],
            "neg_tail_id": neg_tail["input_ids_label"],
            "neg_tail_mask": neg_tail["attention_mask_label"]
        }
        
        
class Distance_predicate(nn.Module, BaseRule):
    def __init__(self, train_path, valid_path, test_path, epoch=10, batch_size =512, early_stop = 5, fine_tune = False, append_path='', current_dir = '', task_name='', device = 'cuda:1', model_name_specific = None, tokenize_length=10, lr = 5e-5, ele_number = 10, final_dim = 10):
        super(Distance_predicate, self).__init__()
        BaseRule.__init__(self, ele_number, final_dim, current_dir, task_name, tokenize_length)
        # import bert and token 
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.early_stop = EarlyStopper(patience=early_stop, min_delta=0.1)
        self.lr = lr
        # self.make_data_cls_embeddings()
        # self.process_example_cls_embeddings()
        # self.tokenize()
        self.device = device
        self.fine_tune = fine_tune
        if model_name_specific == None:
            self.datetime = datetime.datetime.now().strftime("%d%H")
        else: 
            self.datetime = model_name_specific
        if not os.path.exists(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}"):
            os.makedirs(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}")

        self.output_model_path_basic = f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/bert_fine_tune"
        self.output_model_path = self.output_model_path_basic + append_path + '.pt'
    
    def obtain_embedding_bert(self,token, input_id, attention_mask):
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        model.to(self.device)
        model.eval()
        token.set_format('torch', columns=[input_id, attention_mask])
        token_data = torch.utils.data.DataLoader(token, batch_size=32, shuffle=False)
        embeddings_e = []
        for batch in tqdm.tqdm(token_data):
            e_input_id = batch[input_id].to(self.device)
            e_attention_mask = batch[attention_mask].to(self.device)
            e_hidden_states = model(input_ids=e_input_id, attention_mask=e_attention_mask, output_hidden_states=True)
            e_hidden_states = self.get_hidden_states(e_hidden_states, token_ids_word=0)
            embeddings_e.append(e_hidden_states.detach().cpu())
        embeddings_e = torch.cat(embeddings_e, dim=0)
        return embeddings_e
            
    def process_embeddings(self):
        # load the data 
        train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/train_balance_cls_embeddings.csv")['train']
        valid_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/valid_balance_cls_embeddings.csv")['train']
        test_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/test_balance_cls_embeddings.csv")['train']
        # train_data = train_data.select(range(5))
        # valid_data = valid_data.select(range(5))
        # test_data = test_data.select(range(5))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenize_function(examples, main):
            token =  tokenizer(examples[main], return_tensors="pt", padding='max_length', truncation=True, max_length=self.tokenize_length)
            token = {f'input_ids_{main}': token['input_ids'], f'attention_mask_{main}': token['attention_mask']}
            return token
        
        for ds, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            print('transfer', ds)
            token_e = data.map(lambda batch: tokenize_function(batch, 'textE'), batched=True)
            # store the embeddings of all token_e 
            embeddings_e = self.obtain_embedding_bert(token_e, 'input_ids_textE', 'attention_mask_textE')
            torch.save(embeddings_e, f"{self.current_dir}/cache/{self.task_name}/{ds}_balance_bert_embeddings_E_{self.tokenize_length}.pt")
            

            
            token_r = data.map(lambda batch: tokenize_function(batch, 'textR'), batched=True)
            embeddings_r = self.obtain_embedding_bert(token_r, 'input_ids_textR', 'attention_mask_textR')
            torch.save(embeddings_r, f"{self.current_dir}/cache/{self.task_name}/{ds}_balance_bert_embeddings_R_{self.tokenize_length}.pt")
            

            
            token_l = data.map(lambda batch: tokenize_function(batch, 'label'), batched=True)
            embeddings_l = self.obtain_embedding_bert(token_l, 'input_ids_label', 'attention_mask_label')
            torch.save(embeddings_l, f"{self.current_dir}/cache/{self.task_name}/{ds}_balance_bert_embeddings_L_{self.tokenize_length}.pt")

    
    
    def process_example_cls_embeddings(self):
        
        
        # load the data 
        train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/train_balance_cls_embeddings.csv")['train']
        valid_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/valid_balance_cls_embeddings.csv")['train']
        test_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/test_balance_cls_embeddings.csv")['train']
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenize_function(examples, main):
            token =  tokenizer(examples[main], return_tensors="pt", padding='max_length', truncation=True, max_length=self.tokenize_length)
            token = {f'input_ids_{main}': token['input_ids'], f'attention_mask_{main}': token['attention_mask']}
            return token
        
        
        
        for ds, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            print('transfer', ds)
            token_e = data.map(lambda batch: tokenize_function(batch, 'textE'), batched=True)
            token_e.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/{ds}_balance_cls_embeddings_token_E_{self.tokenize_length}.hf")
            
            token_r = data.map(lambda batch: tokenize_function(batch, 'textR'), batched=True)
            token_r.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/{ds}_balance_cls_embeddings_token_R_{self.tokenize_length}.hf")
            
            token_l = data.map(lambda batch: tokenize_function(batch, 'label'), batched=True)
            token_l.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/{ds}_balance_cls_embeddings_token_L_{self.tokenize_length}.hf")
            
    def embedding_head_fine_tune(self):
        # load tensor data
        train_data_E = (f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_E_{self.tokenize_length}.pt")
        train_data_R = (f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_R_{self.tokenize_length}.pt")
        train_data_L = (f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_L_{self.tokenize_length}.pt")
        tuple_data = PNEmbeddings(train_data_E, train_data_R, train_data_L, self.ele_number)
        train_data_con = torch.utils.data.DataLoader(tuple_data, batch_size=self.batch_size, shuffle=False)
        model = BertHead(200)
        model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.lr)
        if self.fine_tune:
            checkpoint = torch.load(self.output_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # criterions = nn.L1Loss()
        criterions = nn.MarginRankingLoss(margin=1000, reduction='mean')
        model.train()
        for single_epoch in range(self.epoch):
            print('Epoch:', single_epoch)
            epoch_loss = 0
            for batch in tqdm.tqdm(train_data_con):
                e = batch["embed_e"].to(self.device)
                r = batch["embed_r"].to(self.device)
                l = batch["embed_l"].to(self.device)
                neg_head = batch["neg_head_embed"].to(self.device)
                neg_tail = batch["neg_tail_embed"].to(self.device)
                e  = model(e)
                r = model(r)
                l = model(l)
                neg_head = model(neg_head)
                neg_tail = model(neg_tail)
                distance = self.get_distance(e, r, l)
                neg_distance = self.get_distance(neg_head, r, neg_tail)
                inverse_distance = self.get_distance(l, r, e)
                optimizer.zero_grad()
                loss = criterions(distance, neg_distance, -1*torch.ones_like(distance, device=self.device))
                inverse_loss = criterions(distance, inverse_distance, -1*torch.ones_like(distance, device=self.device))
                loss = loss + inverse_loss
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()
            epoch_loss = epoch_loss / len(train_data_con)
            print(f"[Train] Epoch: {single_epoch}, Loss: {epoch_loss}")
            self.writer.add_scalar("Loss/train", epoch_loss, single_epoch)
            self.writer.flush()
        self.save(model, optimizer)
    
    def train_with_labels(self):
        # load tensor data
        train_data_E = (f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_E_{self.tokenize_length}.pt")
        train_data_R = (f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_R_{self.tokenize_length}.pt")
        train_data_L = (f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_L_{self.tokenize_length}.pt")
        tuple_data = PNEmbeddings(train_data_E, train_data_R, train_data_L, self.ele_number)
        train_data_con = torch.utils.data.DataLoader(tuple_data, batch_size=self.batch_size, shuffle=False)
        
        # Define model
        model = FactHead(200, self.final_dim)

        model.to(self.device)
        
        optimizer = AdamW(model.parameters(), lr=self.lr)
        if self.fine_tune:
            checkpoint = torch.load(self.output_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterions = nn.L1Loss()
        # criterions = nn.MarginRankingLoss(margin=1000, reduction='mean')
        model.train()
        for single_epoch in range(self.epoch):
            print('Epoch:', single_epoch)
            epoch_loss = 0
            positive_loss = 0
            negative_loss = 0
            for batch, index in tqdm.tqdm(train_data_con):
                e = batch["embed_e"].to(self.device)
                r = batch["embed_r"].to(self.device)
                l = batch["embed_l"].to(self.device)
                neg_head = batch["neg_head_embed"].to(self.device)
                neg_tail = batch["neg_tail_embed"].to(self.device)
                # e  = model(e)
                # r = model(r)
                # l = model(l)
                # neg_head = model(neg_head)
                # neg_tail = model(neg_tail)
                
                positive_predictions = model(e, r, l)
                negative_predictions = model(neg_head, r, neg_tail)
                
                optimizer.zero_grad()
                positive_loss = criterions(positive_predictions, torch.ones_like(positive_predictions, device=self.device))
                negative_loss = criterions(negative_predictions, torch.zeros_like(negative_predictions, device=self.device))
                loss = positive_loss +  negative_loss

                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()
                positive_loss = positive_loss + positive_loss.item()
                negative_loss = negative_loss + negative_loss.item()
            epoch_loss = epoch_loss / len(train_data_con)
            positive_loss = positive_loss / len(train_data_con)
            negative_loss = negative_loss / len(train_data_con)
            print(f"[Train] Epoch: {single_epoch}, Loss: {epoch_loss}, Positive Loss: {positive_loss}, Negative Loss: {negative_loss}")
            self.writer.add_scalar("PretrainLoss/total_loss", epoch_loss, single_epoch)
            self.writer.add_scalar("PretrainLoss/positive", positive_loss, single_epoch)
            self.writer.add_scalar("PretrainLoss/negative", negative_loss, single_epoch)
            self.writer.flush()
        self.save(model, optimizer)
        
        # test the model 
        model.eval()
        tp , tn, fp, fn = 0, 0, 0, 0

        for batch, index in tqdm.tqdm(train_data_con):
            e = batch["embed_e"].to(self.device)
            r = batch["embed_r"].to(self.device)
            l = batch["embed_l"].to(self.device)
            neg_head = batch["neg_head_embed"].to(self.device)
            neg_tail = batch["neg_tail_embed"].to(self.device)
            positive_predictions = model(e, r, l)
            negative_predictions = model(neg_head, r, neg_tail)
            tp = tp + torch.sum(positive_predictions >= 0.5).item()
            fn = fn + torch.sum(positive_predictions < 0.5).item()
            tn = tn + torch.sum(negative_predictions < 0.5).item()
            fp = fp + torch.sum(negative_predictions >= 0.5).item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall  = tp / (tp + fn) if (tp + fn) > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn)
        print(f"[TEST] TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision}, Recall: {recall}, Accuracy: {acc}")
        

    # inference
    def inference_distance(self, model_path = ''):
        infere_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/inference_balance_cls_embeddings.csv")['train']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        def tokenize_function(examples, main):
            token =  tokenizer(examples[main], return_tensors="pt", padding='max_length', truncation=True, max_length=100)
            token = {f'input_ids_{main}': token['input_ids'], f'attention_mask_{main}': token['attention_mask']}
            return token

        test_data_E = infere_data.map(lambda batch: tokenize_function(batch, 'textE'), batched=True)
        test_data_R = infere_data.map(lambda batch: tokenize_function(batch, 'textR'), batched=True)
        test_data_L = infere_data.map(lambda batch: tokenize_function(batch, 'label'), batched=True)

        # load the local data 
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        
        
        head_model = BertHead(200, self.final_dim)
        checkpoint = torch.load(model_path, map_location='cpu')
        head_model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        head_model.to(self.device)
        
        test_data_E.set_format('torch', columns=["input_ids_textE", "attention_mask_textE"])
        test_data_R.set_format('torch', columns=["input_ids_textR", "attention_mask_textR"])
        test_data_L.set_format('torch', columns=["input_ids_label", "attention_mask_label"])
        
        parallel_test = ParallelDataset(test_data_E, test_data_R, test_data_L)
        test_data_con = torch.utils.data.DataLoader(parallel_test, batch_size=1, shuffle=False)
        criterion = nn.L1Loss()
        # test the model
        model.eval()
        epoch_loss_test = 0
        for batch in (test_data_con):
            e_input_id = batch["id_e"].to(self.device)
            e_attention_mask = batch["mask_e"].to(self.device)
            r_input_id = batch["id_r"].to(self.device)
            r_attention_mask = batch["mask_r"].to(self.device)
            l_input_id = batch["id_l"].to(self.device)
            l_attention_mask = batch["mask_l"].to(self.device)
            e_hidden_states = model(input_ids=e_input_id, attention_mask=e_attention_mask, output_hidden_states=True)
            r_hidden_states = model(input_ids=r_input_id, attention_mask=r_attention_mask, output_hidden_states=True)
            l_hidden_states = model(input_ids=l_input_id, attention_mask=l_attention_mask, output_hidden_states=True)
            # embedd the cls token
            e_hidden_states = self.get_hidden_states(e_hidden_states, token_ids_word=0)
            r_hidden_states = self.get_hidden_states(r_hidden_states, token_ids_word=0)
            l_hidden_states = self.get_hidden_states(l_hidden_states, token_ids_word=0)
            # obatin the distance 
            e_hidden_states = head_model(e_hidden_states)
            r_hidden_states = head_model(r_hidden_states)
            l_hidden_states = head_model(l_hidden_states)
            
            distance = self.get_distance(e_hidden_states, r_hidden_states, l_hidden_states)
            loss = criterion(distance, torch.zeros_like(distance))
            # original entity in text
            entity_in_text = tokenizer.decode(e_input_id[0]).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
            relation_in_text = tokenizer.decode(r_input_id[0]).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
            label_in_text = tokenizer.decode(l_input_id[0]).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
            print(f"Entity: {entity_in_text}")
            print(f"Relation: {relation_in_text}")
            print(f"Label: {label_in_text}")
            print(f'Loss: {loss.item()}')
            epoch_loss_test = epoch_loss_test + loss.item()
        epoch_loss_test = epoch_loss_test / len(test_data_con)
        print(f"[Inference] Mean Loss: {epoch_loss_test}")
        
    # save
    def save(self, model, optimizer, append_path=''):
        # save the model 
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, self.output_model_path_basic+append_path+'.pt')
        # check if model number is larger then 4, then delete the oldest file
        model_files = os.listdir(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}")
        if len(model_files) > 4:
            files_with_times = [(self.get_creation_time(f'{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/{x}'), self.get_loss_str(x), x) for x in model_files]
            # find the larger loss 
            files_with_times.sort(key=lambda x: x[0])
            files_with_times.sort(key=lambda x: x[1], reverse=True)
            os.remove(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/{files_with_times[0][2]}")
            
    def get_creation_time(self,file_path):
        """
        Returns the creation timestamp of a file.
        """
        return os.path.getctime(file_path)
    def get_loss_str (self, file_path):
        try:
            file_str = file_path.split('_')
            loss = float(file_str[-1][:-3])
        except:
            loss = 0.0
        return loss
        

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
    def __init__(self, entity_path, relation_path, epoch=10, batch_size =512, early_stop = 5, ele_number=10, current_dir='',device='', task_name='',final_dim = 10,threshold = 1500, train_loop_logic=10, model_name_specific = None, tokenize_length=10, load_pretrain_head=True, lr_predicate_model = 1e-4, lr_rule_model = 1e-3, target_predicate = 'Host a visit', data_format = 'kg'):
        super(DifferentiablePropositionalization, self).__init__()
        BaseRule.__init__(self, ele_number, final_dim, current_dir, task_name,tokenize_length)
        # import bert and token 
        self.data_format =  data_format
        self.entity_path = entity_path
        self.relation_path = relation_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.early_stop = EarlyStopper(patience=early_stop, min_delta=0.1)
        self.number_variable = 3
        self.target_relation = target_predicate
        self.target_arity = 2
        self.device = device
        self.random_negative = 32
        self.bert_dim_length = 768
        self.lr_rule_model = lr_rule_model
        self.lr_predicate_model = lr_predicate_model
        self.threshold = threshold
        self.train_loop_logic = train_loop_logic
        self.rule_path = f"{self.current_dir}/cache/{self.task_name}/rule_output_{self.target_relation}.md"
        self.writer = SummaryWriter(f'{self.current_dir}/cache/{self.task_name}/runs/')
        if model_name_specific == None:
            self.datetime = datetime.datetime.now().strftime("%d%H")
        else: 
            self.datetime = model_name_specific
        if not os.path.exists(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}"):
            os.makedirs(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}")
        self.model_path_predicate_path = f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/bert_fine_tune.pt"
        self.output_model_path_basic = f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/soft_proposition"
        self.variable_perturbations_list = list(itertools.permutations(range(self.number_variable), 2))
        self.mapping = {0:'X', 1:'Y', 2:'Z'}
        self.left_bracket = '['
        self.right_bracket = ']'
        self.split = '@'
        self.end = '#'
        self.get_index_X_Y = self.variable_perturbations_list.index((0,1))
        # self.variable_perturbations.remove((0,1))
        self.number_variable_terms  = len(self.variable_perturbations_list)
        self.variable_perturbations = torch.tensor(self.variable_perturbations_list, device=self.device)
        self.load_pretrain_head = load_pretrain_head
        # write the pertumation to the file
        with open(f"{self.current_dir}/cache/{self.task_name}/variable_perturbations.txt", "w") as f:
            f.write(str(self.variable_perturbations))
            f.close()
        
        if self.data_format == 'image':
            self.n_clusters = 10

    def make_tokeniz(self):
        '''
        # ! Run this function in the first time 
        '''
        # read train data and generated entity and realtison 
        if self.data_format == 'kg':
            train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/train_balance_cls_embeddings.csv")['train']
        elif self.data_format == 'image':
            train_data = load_dataset('csv', data_files = f"{self.current_dir}/cache/{self.task_name}/embeddings.csv")['train']
        else:
            raise('The data format need to correct')
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
    
    # def soft_pro(self):
    #     '''
    #     Read all entities and process and propositionalalization and rule learning task together 
    #     '''
    #     # read all entities 
    #     entity_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/entity_token.hf")
    #     relation_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/relation_token.hf")
    #     self.tar_predicate_index = None
    #     number_relations = len(relation_data)
    #     self.all_relations = relation_data['text']
    #     print('All relations:', self.all_relations) 
        
    #     self.all_unground_atoms = []
    #     for arrage in self.variable_perturbations_list:
    #         for item in self.all_relations:
    #             self.all_unground_atoms.append(f"{item}[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")
    #     target_index = self.all_unground_atoms.index(f'{self.target_relation}[X@Y]')
    #     for item in range(number_relations):
    #         if relation_data[item]['text'] == self.target_relation.replace('_', ' '):
    #             self.tar_predicate_index = item
    #             break
    #     self.target_atom_index = number_relations * self.get_index_X_Y + self.tar_predicate_index
    #     self.target_atom_index = self.get_index_X_Y * self.number_variable_terms + self.target_atom_index
    #     assert self.target_atom_index == target_index, f"Target atom index {self.target_atom_index} is not equal to the target index {target_index}"
    #     entity_data.set_format('torch', columns=["input_ids", "attention_mask"])
    #     relation_data.set_format('torch', columns=["input_ids", "attention_mask"])


    #     #load the data and neural predicate model
    #     substation_data = SubData(entity_data, self.number_variable, self.train_loop_logic)
    #     substation_data = torch.utils.data.DataLoader(substation_data, batch_size=1, shuffle=False)

    #     predicate = BertForMaskedLM.from_pretrained('bert-base-uncased')
    #     #! load pretrain weights 
    #     head_model = BertHead(200)
    #     checkpoint = torch.load(self.model_path_predicate_path, map_location='cpu')
    #     head_model.load_state_dict(checkpoint['model_state_dict'])
    #     # predicate.load_state_dict(torch.load(f"{self.current_dir}/cache/{self.task_name}/out/0106/bert_fine_tune_balance_0.0.pt")['model_state_dict'])
    #     predicate.to(self.device)
    #     head_model.to(self.device)
        
    #     # Load neural logic rule learning model 
    #     self.logic_model = DeepRuleLayer(number_relations*self.number_variable_terms-1, rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=self.rule_path, t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=self.target_arity,time_stamp='' )
    #     self.logic_model.to(self.device)
        
    #     # define optimizer to train logic and predicate models 

    #     # Define parameter groups
    #     parameter_groups = [
    #         {'params': self.logic_model.parameters(), 'lr': 1e-3},  # Lower learning rate for the first layer
    #         {'params': head_model.parameters(), 'lr': 1e-2}  # Higher learning rate for the second layer
    #     ]

    #     logic_optimizer = torch.optim.Adam(parameter_groups)

    #     criterions = nn.MSELoss()

    #     relation_data = DataLoader(relation_data, batch_size=len(relation_data), shuffle=False)
    #     for batch_index, single_bath in enumerate(relation_data):
    #         if batch_index > 0:
    #             raise ValueError("Only one batch is allowed")
    #         single_bath = {k: v.to(self.device) for k, v in single_bath.items()}
    #         all_r_embeddings = predicate(input_ids=single_bath['input_ids'], attention_mask=single_bath['attention_mask'], output_hidden_states=True)
    #         all_r_embeddings = self.get_hidden_states(all_r_embeddings, token_ids_word=0)
    #         all_r_embeddings = head_model(all_r_embeddings)
        
    #     all_r_embeddings = all_r_embeddings.unsqueeze(0)
    #     all_r_embeddings  = all_r_embeddings.expand(self.number_variable_terms, -1, -1)
    #     all_r_embeddings = all_r_embeddings.unsqueeze(2)
    #     # start training
    #     self.logic_model.train()
    #     rule_set = None
    #     for single_epoch in range(self.epoch):
    #         for batch_index in tqdm.tqdm(substation_data):
    #             # get the embedding of the entity 
    #             batch, index = batch_index
    #             index_cpu = list(index.detach().cpu().numpy()[0])
    #             # print(entity_data[int(index_cpu[0])], entity_data[int(index_cpu[1])], entity_data[int(index_cpu[2])])
    #             #untoken
    #             index = index.to(self.device)
    #             index = index.squeeze(0)
    #             first_entity_id = batch['first_entity_id'].to(self.device)  # obj entity 
    #             first_entity_mask = batch['first_entity_mask'].to(self.device)
    #             second_entity_id = batch['second_entity_id'].to(self.device) # label entity / sub entity / second entity 
    #             second_entity_mask = batch['second_entity_mask'].to(self.device)
    #             third_entity_id = batch['third_entity_id'].to(self.device)
    #             third_entity_mask = batch['third_entity_id'].to(self.device)
                
    #             # get the distance between the entity and relation
    #             first_output = predicate(input_ids=first_entity_id, attention_mask=first_entity_mask, output_hidden_states=True)
    #             second_output = predicate(input_ids=second_entity_id, attention_mask=second_entity_mask, output_hidden_states=True)
    #             third_output = predicate(input_ids=third_entity_id, attention_mask=third_entity_mask, output_hidden_states=True)
    #             # embedd the cls token
    #             first_embeddings = self.get_hidden_states(first_output, token_ids_word=0)
    #             second_embeddings = self.get_hidden_states(second_output, token_ids_word=0)
    #             third_embeddings = self.get_hidden_states(third_output, token_ids_word=0)
    #             first_embeddings = head_model(first_embeddings)
    #             second_embeddings = head_model(second_embeddings)
    #             third_embeddings = head_model(third_embeddings)
            
                
    #             # obtain the substitution for all atoms in body 
    #             # get the all variable perturbation and return the substitutions based on X, Y, and Z variable
    #             # self.variable_perturbations = self.variable_perturbations.unsqueeze(0).expand(self.batch_size, -1, -1)
    #             #! todo this can be a graph in the pp
    #             perturbations = index[self.variable_perturbations]
    #             # first list 
    #             perturbations = perturbations.unsqueeze(-1).expand(-1,-1,self.final_dim)
    #             first_index = index[0].unsqueeze(-1).expand(self.final_dim)
    #             perturbations = torch.where(perturbations == first_index, first_embeddings, perturbations)
    #             perturbations = torch.where(perturbations == index[1].unsqueeze(-1).expand(self.final_dim), second_embeddings, perturbations)
    #             perturbations = torch.where(perturbations == index[2].unsqueeze(-1).expand(self.final_dim), third_embeddings, perturbations)

    #             # combine the entity and relation
    #             perturbations = perturbations.unsqueeze(1)
    #             expanded_per = perturbations.expand(-1, number_relations, -1, -1)

    #             # include the relations 
    #             triples = torch.cat([expanded_per, all_r_embeddings], dim=2)        
    #             triples = triples.reshape(-1, 3, self.final_dim) # R_0(X,Y), R_1(X,Y), ...,R_n(X,Y) ,R_0(Y,X),.., R_n(Y,X), ... X (e1, e2, r) X embedding length 
    #             # replace elements to embeddings )
                

    #             # get the distance between the entity and relation
    #             e_hidden_states = triples[:,0,:]
    #             r_hidden_states = triples[:,2,:]
    #             l_hidden_states = triples[:,1,:]
    #             distance = self.get_distance(e_hidden_states, r_hidden_states, l_hidden_states)
    #             max_distance = distance.max()
    #             min_distance = distance.min()  
    #             # use the mean of max min as the threshold
    #             # threshold = (max_distance - min_distance)/2
    #             threshold = self.threshold
    #             body_predicate_labels_predicated = torch.where(distance - threshold < 0, torch.ones_like(distance), torch.zeros_like(distance))
    #             head_predicate_labels_predicated = body_predicate_labels_predicated[self.target_atom_index]
    #             if head_predicate_labels_predicated.item() == 1:
    #                 first_entity_id_cpu = first_entity_id[0].detach().cpu().numpy()
    #                 second_entity_id_cpu = second_entity_id[0].detach().cpu().numpy()
    #                 original_x = self.tokenizer.decode(first_entity_id_cpu).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
    #                 original_y = self.tokenizer.decode(second_entity_id_cpu).replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
    #                 print(f"Original X: {original_x}")
    #                 print(f"Original Y: {original_y}")
    #                 # todo: there are multiple false alarm here, try to finetune the welltrained distance model here to retrain the model. The loss include two, first is a constractive loss, the second is a human defined loss which assume the distance of positive examples is less than 5000 and the distance of the negative exampels is larger than 5000 
    #                 if 'Korea' in original_x and 'Korea' in original_y:
    #                     time.sleep(1)
    #             # print(head_predicate_labels_predicated)
    #             body_predicate_labels_predicated = torch.cat([body_predicate_labels_predicated[:self.target_atom_index], body_predicate_labels_predicated[self.target_atom_index+1:]])
    #             # print(body_predicate_labels_predicated)
    #             # find the target order of X and Y 
                
    #             # append rule learning module 
    #             predicated = self.logic_model(body_predicate_labels_predicated)
    #             loss = criterions(predicated, head_predicate_labels_predicated)
    #             logic_optimizer.zero_grad()
    #             loss.backward()
    #             logic_optimizer.step()
    #             # print the loss
    #             print(f"Epoch: {single_epoch}, Loss: {loss.item()}")
    #             self.writer.add_scalar("Loss/train_Logic", loss.item(), single_epoch)
    #             self.writer.flush()
    #             # if self.early_stop.early_stop(loss.item()):
    #             #     print("[Early Stopping Training]")
    #             #     break
    #         self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
    #         acc, rule_set = self.check_acc(rule_set)
    #         # save rule set 
    #         with open(self.rule_path+'.pk', "wb") as f:
    #             pickle.dump(rule_set, f)
    #             f.close()
    #         precision = acc['precision']
    #         recall = acc['recall']
    #         if recall > 0.5:
    #             break
    #     print("Training finished")
    #     print(f"Precision: {precision}, Recall: {recall}")
    #     return 0 
    
    
    
    def rule_body_train(self):
        '''
        Read all entities and process and propositionalalization and rule learning task together.
        Train distance and logic model together.
        BERT embedding is fix but head embeddings is trainable.
        '''
        # read all entities 
        if self.data_format == 'kg':
            train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/train_balance_cls_embeddings.csv")['train']
        elif self.data_format == 'image':
            train_data = load_dataset('csv',  data_files=f"{self.current_dir}/cache/{self.task_name}/embeddings.csv")['train']
            # get all text 1
            all_label1 = [i for i in train_data['text1']]
            all_label2 = [ i for i in train_data['text2']]
            self.all_entity_labels = all_label1 + all_label2
            all_relations_in_order = [i for i in train_data['textR']]
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


        self.all_unground_atoms = []
        for arrage in self.variable_perturbations_list:
            for item in self.all_relations:
                self.all_unground_atoms.append(f"{item}[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")
        target_index = self.all_unground_atoms.index(f'{self.target_relation}[X@Y]')
        for item in range(number_relations):
            if relation_data[item]['text'] == self.target_relation:
                self.tar_predicate_index = item
                break
        self.target_atom_index = number_relations * self.get_index_X_Y + self.tar_predicate_index
        self.target_atom_index = self.get_index_X_Y * self.number_variable_terms + self.target_atom_index
        assert self.target_atom_index == target_index, f"Target atom index {self.target_atom_index} is not equal to the target index {target_index}"
                
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
        tuple_data = BatchZ(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index)
        substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=False)
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
        self.logic_model = DeepRuleLayer(number_relations*self.number_variable_terms-1, rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=self.rule_path, t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=self.target_arity,time_stamp='')
        self.logic_model.to(self.device)
        
        # Define the differentiable k-means module for image task
        if self.data_format == 'image':
            self.cluster = DkmCompGraph(ae_specs=self.bert_dim_length, n_clusters=self.n_clusters, val_lambda=1, input_size=None, device=self.device, mode='without_vae')
            # with all entity embeddings from image for initialization all_obj_bert_embeddings, compute the centeriod with trasitional kmeans 
            all_obj_bert_embeddings_numpy = all_obj_bert_embeddings.cpu().numpy()
            print('[pretraining done]')
            kmeans_model = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(all_obj_bert_embeddings_numpy)
            print('[kmeans done]')
            print('[K-Means prediction for embeddings]')
            print(kmeans_model.predict(all_obj_bert_embeddings_numpy))
            print('[Labels]')
            print(self.all_entity_labels)
            acc_initial = accuracy_score(kmeans_model.labels_, self.all_entity_labels)
            print(f'[Accuracy of initial K-Means]')
            print(acc_initial)
            self.cluster.cluster_rep = torch.nn.Parameter(torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32, device=self.device), requires_grad=True)


        # Define parameter groups
        if self.data_format == 'image':
            parameter_groups = [
                {'params': self.logic_model.parameters(), 'lr': 1e-3},  # Lower learning rate for the first layer
                {'params': head_model.parameters(), 'lr': 1e-4},  # Higher learning rate for the second layer
                {'params': self.cluster.parameters(), 'lr': 1e-4}  # Higher learning rate for the second layer
            ]
        else:
            parameter_groups = [
                {'params': self.logic_model.parameters(), 'lr': 1e-3},  # Lower learning rate for the first layer
                {'params': head_model.parameters(), 'lr': 1e-4}  # Higher learning rate for the second layer
            ]
        logic_optimizer = torch.optim.Adam(parameter_groups)


        # start training
        self.logic_model.train()
        head_model.train()
        rule_set = None
        for single_epoch in range(self.epoch):
            total_loss_rule = 0
            total_loss = 0
            total_dkm_loss = 0
            total_body_loss = 0
            for index in tqdm.tqdm(substation_data):
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
                # check the occurence 
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
                body_loss = torch.nn.BCELoss()(ground_truth_values, labels_body) # can use Cross Entropy Loss
                
                #! the threshold is human-defined here now 
                # ! Do not binaries the values for the rule learning model 
                # threshold = 0.5
                # ground_truth_values = torch.where(ground_truth_values > threshold , torch.ones_like(ground_truth_values), torch.zeros_like(ground_truth_values))
                head_predicate_labels_predicated = ground_truth_values[:,self.target_atom_index].unsqueeze(1)
                body_predicate_labels_predicated = torch.cat([ground_truth_values[:,:self.target_atom_index], ground_truth_values[:,self.target_atom_index+1:]],dim=1)
                # print(body_predicate_labels_predicated)
                # print(body_predicate_labels_predicated.max(), body_predicate_labels_predicated.min())
                # # print the index when body is zero 
                # indices = torch.nonzero(body_predicate_labels_predicated == 1  , as_tuple=False)
                # print("Indices:", indices)
                # append rule learning module 
                predicated = self.logic_model(body_predicate_labels_predicated)
                loss_rule = torch.nn.MSELoss()(predicated, head_predicate_labels_predicated)

                #loss for dkm 
                if self.data_format == 'image':
                    dkm_loss = l1 + l2
                else: 
                    dkm_loss = 0
                
                # all loss from logic rule loss, body predicate, and [deep k-means]
                loss = 0.5*loss_rule + 0.5*body_loss + 0.5 * dkm_loss
                
                logic_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # todo define the loss from neural predicate here 
                logic_optimizer.step()
                total_loss_rule = total_loss_rule + loss_rule.item()
                total_loss += loss.item()
                total_body_loss += body_loss.item()
                total_dkm_loss += dkm_loss.item()

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
            if recall > 0.5:
                break
        self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
        print("Training finished")
        print(f"Precision: {precision}, Recall: {recall}")
        return 0 

    def build_kg_based_on_cluster(self, selected_train_E, selected_train_L, all_relations):
        first_labels = self.cluster.get_cluster_index(selected_train_E)
        second_labels = self.cluster.get_cluster_index(selected_train_L)
        all_predicted_labels = list(first_labels) + list(second_labels)
        acc_dkm = accuracy_score(all_predicted_labels, self.all_entity_labels)
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

    def soft_pro_full_end(self):
        '''
        Read all entities and process and propositionalalization and rule learning task together.
        Train distance and logic model together.
        BERT embedding is fix but head embeddings is trainable.
        '''
        # read all entities 
        train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/train_balance_cls_embeddings.csv")['train']
        # entity_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/entity_token.hf")
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
        # compute all relation bert embeddings 
        predicate = BertForMaskedLM.from_pretrained('bert-base-uncased')
        predicate.to(self.device)

        self.all_unground_atoms = []
        for arrage in self.variable_perturbations_list:
            for item in self.all_relations:
                self.all_unground_atoms.append(f"{item}[{self.mapping[arrage[0]]}@{self.mapping[arrage[1]]}]")
        target_index = self.all_unground_atoms.index(f'{self.target_relation}[X@Y]')
        for item in range(number_relations):
            if relation_data[item]['text'] == self.target_relation.replace('_', ' '):
                self.tar_predicate_index = item
                break
        self.target_atom_index = number_relations * self.get_index_X_Y + self.tar_predicate_index
        self.target_atom_index = self.get_index_X_Y * self.number_variable_terms + self.target_atom_index
        assert self.target_atom_index == target_index, f"Target atom index {self.target_atom_index} is not equal to the target index {target_index}"
        relation_data.set_format('torch', columns=["input_ids", "attention_mask"])

        relation_data_embedding = DataLoader(relation_data, batch_size=len(relation_data), shuffle=False)
        for batch_index, single_bath in enumerate(relation_data_embedding):
            if batch_index > 0:
                raise ValueError("Only one batch is allowed")
            single_bath = {k: v.to(self.device) for k, v in single_bath.items()}
            all_relation_bert_embedding = predicate(input_ids=single_bath['input_ids'], attention_mask=single_bath['attention_mask'], output_hidden_states=True)
            all_relation_bert_embedding = self.get_hidden_states(all_relation_bert_embedding, token_ids_word=0)
        
        # load the positive and negative data and an random index 
        train_data_E = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_E_{self.tokenize_length}.pt")
        train_data_R = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_R_{self.tokenize_length}.pt")
        train_data_L = torch.load(f"{self.current_dir}/cache/{self.task_name}/train_balance_bert_embeddings_L_{self.tokenize_length}.pt")
        
        selected_train_E = train_data_E[:self.ele_number].to(self.device)
        selected_train_L = train_data_L[:self.ele_number].to(self.device)
        selected_train_R = train_data_R[:self.ele_number].to(self.device)
        unique_relation_bert_embeddings = torch.unique(selected_train_R, dim=0)
        # prepare build all KB with bert embedding 
        all_facts_bert_embeddings = torch.cat([selected_train_E, selected_train_R, selected_train_L], dim=1) # get all true index of fact
        all_obj_bert_embeddings = torch.cat([selected_train_E, selected_train_L], dim=0) # get all true index of object
        selected_r_bert_embeddings = all_relation_bert_embedding.unsqueeze(0)
        selected_r_bert_embeddings  = selected_r_bert_embeddings.expand(self.number_variable_terms, -1, -1)
        selected_r_bert_embeddings = selected_r_bert_embeddings.unsqueeze(2)
        # define the data 
        tuple_data = BatchNegative(train_data_E, train_data_R, train_data_L, self.ele_number, random_negative = self.random_negative, target_atoms_index=target_atoms_index)
        substation_data = torch.utils.data.DataLoader(tuple_data, batch_size=1, shuffle=False)
        # load pretrain weights 
        head_model = FactHead(200)
        checkpoint = torch.load(self.model_path_predicate_path, map_location='cpu')
        head_model.load_state_dict(checkpoint['model_state_dict'])
        head_model.to(self.device)
        
        # Load neural logic rule learning model 
        self.logic_model = DeepRuleLayer(number_relations*self.number_variable_terms-1, rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=self.rule_path, t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=self.target_arity,time_stamp='')
        self.logic_model.to(self.device)
        
        # Define parameter groups
        parameter_groups = [
            {'params': self.logic_model.parameters(), 'lr': 1e-2},  # Lower learning rate for the first layer
            {'params': head_model.parameters(), 'lr': 1e-1}  # Higher learning rate for the second layer
        ]
        logic_optimizer = torch.optim.Adam(parameter_groups)
        criterions = nn.MSELoss()
        # start training
        self.logic_model.train()
        head_model.train()
        rule_set = None
        for single_epoch in range(self.epoch):
            total_loss_predicate = 0 
            total_loss_predicate_positive = 0
            total_loss_predicate_negative = 0
            total_loss_rule = 0
            total_loss = 0
            total_body_loss = 0
            for batch_index in tqdm.tqdm(substation_data):
                # get the head embedding of the entity 
                
                selected_train_E_head_embeddings = head_model.generate_embedding(selected_train_E)
                selected_train_L_head_embeddings = head_model.generate_embedding(selected_train_L)
                selected_r_head_embeddings = head_model.generate_embedding(all_relation_bert_embedding)
                all_objects_head_embedding = torch.cat([selected_train_E_head_embeddings, selected_train_L_head_embeddings], dim=0).to(self.device)
                
                selected_r_head_embeddings = selected_r_head_embeddings.unsqueeze(0)
                selected_r_head_embeddings  = selected_r_head_embeddings.expand(self.number_variable_terms, -1, -1)
                selected_r_head_embeddings = selected_r_head_embeddings.unsqueeze(2)
                
                batch, index = batch_index
                # print(entity_data[int(index_cpu[0])], entity_data[int(index_cpu[1])], entity_data[int(index_cpu[2])])
                #untoken
                index = index.to(self.device)
                index = index.squeeze(0)

                first_embeddings_bert = batch["embed_e"].to(self.device)
                second_embeddings_bert = batch["embed_l"].to(self.device)
                third_embeddings_bert = batch["random_z_obj"].to(self.device)
                relation_embedding_bert = batch["embed_r"].to(self.device)
                
                neg_fir_embedding_bert = batch["neg_head_embed"].to(self.device).squeeze(0)
                neg_sec_embedding_bert = batch["neg_tail_embed"].to(self.device).squeeze(0)
                batch_relation_embedding_bert = relation_embedding_bert.expand(neg_fir_embedding_bert.shape[0], -1)
                
                # # ! We only consider the X Y Z three variables in this stage
                # first_embeddings = head_model.generate_embedding(first_embeddings_bert)
                # second_embeddings = head_model.generate_embedding(second_embeddings_bert)
                # third_embeddings = head_model.generate_embedding(third_embeddings_bert)

                positive_values = head_model(first_embeddings_bert, relation_embedding_bert, second_embeddings_bert)
                loss_predicate_positive =  criterions(positive_values, torch.ones_like(positive_values))
                
                negative_values = head_model(neg_fir_embedding_bert, batch_relation_embedding_bert, neg_sec_embedding_bert) # based on the target relation, generate some negative exmaples 
                loss_predicate_negative =  criterions(negative_values, torch.zeros_like(negative_values))
                
                
                # obtain the substitution for all atoms in body 
                # get the all variable perturbation and return the substitutions based on X, Y, and Z variable
                # self.variable_perturbations = self.variable_perturbations.unsqueeze(0).expand(self.batch_size, -1, -1)
                all_row = torch.arange(0, index.shape[0], device=self.device).unsqueeze(1).expand(-1, self.variable_perturbations.shape[1])
                all_row = all_row.unsqueeze(1).expand(-1, self.variable_perturbations.shape[0],-1)
                variable_perturbations_variance = self.variable_perturbations.unsqueeze(0).expand(index.shape[0], -1, -1)
                perturbations = index[all_row, variable_perturbations_variance]
                
                # first find labels for all perturbation 
                #! Based on the all body predicates and entity pairs, generate the labels 
                perturbations_bert_embeddings = all_obj_bert_embeddings[perturbations]
                perturbations_bert_embeddings = perturbations_bert_embeddings.unsqueeze(2)
                expanded_per = perturbations_bert_embeddings.expand(-1, -1, number_relations, -1, -1)
                selected_r_bert_embeddings_var = selected_r_bert_embeddings.unsqueeze(0).expand(expanded_per.shape[0], -1, -1, -1, -1)
                triples = torch.cat([expanded_per, selected_r_bert_embeddings_var], dim=3)
                triples = triples.reshape(self.random_negative, -1, 3, self.bert_dim_length)
                triples = triples.reshape(-1, 3, self.bert_dim_length)
                first_shape = triples.shape[0]
                triples = triples.reshape(first_shape,-1)
                # check the occurence 
                triples = triples.unsqueeze(1)
                all_facts_bert_embeddings_var = all_facts_bert_embeddings.unsqueeze(0)
                labels_body = (triples==all_facts_bert_embeddings_var).all(dim=2).any(dim=1)
                labels_body = labels_body.reshape(self.random_negative, -1).float()
                                
                # get the embeddings for all perturbations 
                perturbations_embeddings = all_objects_head_embedding[perturbations]
                perturbations_embeddings = perturbations_embeddings.unsqueeze(2)
                expanded_per = perturbations_embeddings.expand(-1, -1, number_relations, -1, -1)
                all_r_embeddings_variance = selected_r_head_embeddings.unsqueeze(0).expand(expanded_per.shape[0], -1, -1, -1, -1)
                triples = torch.cat([expanded_per, all_r_embeddings_variance], dim=3)    
                triples = triples.reshape(self.random_negative, -1, 3, self.final_dim) # R_0(X,Y), R_1(X,Y), ...,R_n(X,Y) ,R_0(Y,X),.., R_n(Y,X), ... X (e1, e2, r) X embedding length 
                triples = triples.reshape(-1, 3, self.final_dim)
                
                # get the distance between the entity and relation
                e_hidden_states = triples[:,0,:]
                r_hidden_states = triples[:,2,:]
                l_hidden_states = triples[:,1,:]
                ground_truth_values = head_model.get_possible(e1=e_hidden_states, embed_r= r_hidden_states, e2= l_hidden_states)
                ground_truth_values = ground_truth_values.reshape(self.random_negative, -1)

                body_loss = criterions(ground_truth_values, labels_body)
                #! the threshold is human-defined here now 
                threshold = 0.5
                body_predicate_labels_predicated = torch.where(ground_truth_values > threshold , torch.ones_like(ground_truth_values), torch.zeros_like(ground_truth_values))
                head_predicate_labels_predicated = body_predicate_labels_predicated[:,self.target_atom_index].unsqueeze(1)
                # if head_predicate_labels_predicated.item() == 1:
                #     original_x = train_data['textE'][int(index_cpu[0])]
                #     original_y = train_data['label'][int(index_cpu[1])]
                #     if index_cpu[2] < self.ele_number:
                #         original_z = train_data['textE'][int(index_cpu[2])]
                #     else:
                #         random_z = int(index_cpu[2]) - self.ele_number   
                #         original_z = train_data['label'][random_z]
                        
                #     print(f"Original X: {original_x}")
                #     print(f"Original Y: {original_y}")
                #     print(f"Random Z: {original_z}")
                #     # todo: there are multiple false alarm here, try to finetune the welltrained distance model here to retrain the model. The loss include two, first is a constractive loss, the second is a human defined loss which assume the distance of positive examples is less than 5000 and the distance of the negative exampels is larger than 5000 
                #     if 'Korea' in original_x and 'Korea' in original_y:
                #         time.sleep(1)
                # print(head_predicate_labels_predicated)
                body_predicate_labels_predicated = torch.cat([body_predicate_labels_predicated[:,:self.target_atom_index], body_predicate_labels_predicated[:,self.target_atom_index+1:]],dim=1)
                
                # todo print the predicted values from neural predicate
                # print(body_predicate_labels_predicated)
                # print(body_predicate_labels_predicated.max(), body_predicate_labels_predicated.min())
                # append rule learning module 
                predicated = self.logic_model(body_predicate_labels_predicated)
                loss_rule = criterions(predicated, head_predicate_labels_predicated)
                loss_predicate =  0.1 * loss_predicate_positive + 0.9 * loss_predicate_negative
                # loss = 0.2*loss_rule + 0.8*loss_predicate # loss = 0.5 * loss_rule + 0.5 * loss_predicate
                loss = 0.2*loss_rule + 0.8*body_loss    
                logic_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # todo define the loss from neural predicate here 
                logic_optimizer.step()
                total_loss_predicate_positive = total_loss_predicate_positive + loss_predicate_positive.item()
                total_loss_predicate_negative = total_loss_predicate_negative + loss_predicate_negative.item()
                total_loss_predicate = total_loss_predicate + loss_predicate.item()
                total_loss_rule = total_loss_rule + loss_rule.item()
                total_loss += loss.item()
                total_body_loss += body_loss.item()

            total_loss_predicate = total_loss_predicate / len(substation_data)
            total_loss_rule = total_loss_rule / len(substation_data)
            total_loss = total_loss / len(substation_data)
            total_loss_predicate_positive = total_loss_predicate_positive / len(substation_data)
            total_loss_predicate_negative = total_loss_predicate_negative / len(substation_data)
            total_body_loss = total_body_loss / len(substation_data)
            # print(f"Epoch: {single_epoch}, Loss: {total_loss}, Predicate Loss: {total_loss_predicate}, Rule Loss: {total_loss_rule}, Positive Loss: {total_loss_predicate_positive}, Negative Loss: {total_loss_predicate_negative}")
            print(f"Epoch: {single_epoch}, Loss: {total_loss}, Predicate Loss: {total_body_loss}, Rule Loss: {total_loss_rule}")
            self.writer.add_scalar("Loss/Logic_predicate_positive", total_loss_predicate_positive, single_epoch)
            self.writer.add_scalar("Loss/Logic_predicate_negative", total_loss_predicate_negative, single_epoch)  
            self.writer.add_scalar("Loss/train_Logic", total_loss_rule, single_epoch)
            self.writer.add_scalar("Loss/train_Predicate", total_loss_predicate, single_epoch)
            self.writer.add_scalar("Loss/train", total_loss, single_epoch)
            self.writer.add_scalar("Loss/body_loss", total_body_loss, single_epoch)
            self.writer.flush()

            acc, rule_set = self.check_acc(rule_set)
            # save rule set 
            with open(self.rule_path+'.pk', "wb") as f:
                pickle.dump(rule_set, f)
                f.close()
            precision = acc['precision']
            recall = acc['recall']
            if recall > 0.5:
                break
        self.save(self.logic_model, logic_optimizer, append_path=f"_logic_")
        print("Training finished")
        print(f"Precision: {precision}, Recall: {recall}")
        return 0 
    
    def build_target_validation_data(self, fact_set):
        with open(f'{self.current_dir}/cache/{self.task_name}/{self.move_special_character(self.target_relation)}.nl','w') as f:
            for item in fact_set:
                first_entity = self.move_special_character(item['textE'])
                relation = self.move_special_character(item['textR'])
                second_entity = self.move_special_character(item['label'])
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
        for item in self.all_unground_atoms:
            if item == f'{self.target_relation}[X@Y]':
                continue
            unground.append(self.move_special_character(item))
        rule_set = self.logic_model.interpret(unground, existing_rule_set= existing_rule_set, scale=False)
        # save realtions
        # print rules 
        with open(self.rule_path, "w") as f:
            print(rule_set,file=f)
            f.close()
        # sort all test facts into a file 
        format_all_relation = []
        for item in self.all_relations:
            format_all_relation.append(self.move_special_character(item))
        # check the accuracy
        KG_checker = CheckMetrics(t_relation=self.move_special_character(self.target_relation), task_name = self.task_name, logic_path=self.rule_path, ruleset = rule_set, t_arity=self.target_arity, data_path = f'{self.current_dir}/cache/{self.task_name}/',all_relation=format_all_relation)
        # check the accuracy
        acc = KG_checker.check_correctness_of_logic_program(left_bracket=self.left_bracket, right_bracket=self.right_bracket, split=self.split, end_symbol=self.end, split_atom='@', left_atom='[', right_atom=']')  
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
        
    def variable_per_layer(self, entity_list):
        # read all entities from entity list 
        
        
        
        return 0

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






def pre_train_predicate_neural(args):
    current_dir = args.folder_name
    task_name = args.task
    data_info = 'balance'
    train_path = f"{current_dir}/cache/{task_name}/triple_train_balance.csv"
    valid_path = f"{current_dir}/cache/{task_name}/triple_train_balance.csv"
    test_path = f"{current_dir}/cache/{task_name}/triple_train_balance.csv"
    
    model = Distance_predicate(train_path, valid_path, test_path, epoch=args.epoch, batch_size =args.batch_size, fine_tune = args.fine_tune, model_name_specific=args.model_path,current_dir=current_dir, task_name=task_name, early_stop=args.early_stop, device=args.device, tokenize_length = args.tokenize_length,lr=args.lr, ele_number=args.element_number)
    # model.process_embeddings()
    if args.inf == False:
        # model.embedding_head_fine_tune()
        model.train_with_labels()
    else:
        model.inference_distance(model_path = f'{args.model_path_parent}/{args.model_path}/bert_fine_tune.pt')

def soft_pro_train_rules(args):
# if __name__ == "__main__":
    current_dir = args.folder_name
    task_name = args.task
    entity_path = f"{current_dir}/cache/{task_name}/all_entities_sampled.csv" # all entities from train file
    relation_path = f"{current_dir}/cache/{task_name}/all_relations_sampled.csv" # all relations from train file
    torch.manual_seed(0)
    model = DifferentiablePropositionalization(entity_path, relation_path, epoch=args.logic_epoch, batch_size =2, current_dir=args.folder_name, device=args.device,task_name = args.task, threshold=5000, train_loop_logic=args.train_loop_logic, model_name_specific=args.model_path, ele_number=args.element_number, tokenize_length=args.tokenize_length, load_pretrain_head=args.load_pretrain_head, lr_predicate_model=args.lr_predicate, lr_rule_model=args.lr_rule, target_predicate=args.target_predicate, data_format=args.data_format)
    model.make_tokeniz()
    # model.soft_pro()
    # model.soft_pro_full_end()
    model.rule_body_train()
    model.only_check_precision_recall_rules()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000, help='number of epochs to train neural predicates')
    parser.add_argument("--logic_epoch", type=int, default=20, help='number of epochs to train neural predicates and logics')
    parser.add_argument("--batch_size", type=int, default=5096)
    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--folder-name", type=str, default=f'{os.getcwd()}/gammaILP/')
    parser.add_argument('--model-path-parent', type=str, default=f'{os.getcwd()}/gammaILP/cache/icews14/out/')
    parser.add_argument('--model-path', type=str, default='head_only_test_demo_l1', help='specific model path to load the model')
    parser.add_argument('--tokenize_length', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_predicate', type=float, default=1e-4)
    parser.add_argument('--lr_rule', type=float, default=1e-3)
    parser.add_argument('--inf', action='store_true')
    parser.add_argument('--train_loop_logic', type=int, default=1000, help='number of train loop for logic module')
    parser.add_argument('--load_pretrain_head', action='store_true')
    parser.add_argument('--target_predicate', type=str, default='lessthan', help='target predicate')
    parser.add_argument("--task", type=str, default='mnist')
    parser.add_argument('--data_format', type=str, default='image', help='Can be chosen from kg or image')
    parser.add_argument('--element_number', type=int, default= -1, help='number of elements to sample for training, if the number is less than 0, then the number of elements is the number of all elements')
    # parser.add_argument('--target_predicate', type=str, default='Host a visit', help='target predicate')
    # parser.add_argument("--task", type=str, default='icews14')
    # parser.add_argument('--data_format', type=str, default='kg', help='Can be chosen from kg or image')
    # parser.add_argument('--element_number', type=int, default= 100, help='number of elements to sample for training')
    args = parser.parse_args()
    # record the parameters 
    print(args)
    torch.manual_seed(1)
    # pre_train_predicate_neural(args)
    soft_pro_train_rules(args)



# todo The problem may appear at two stage:
# todo 1. The distance predicate is not accurate 
# todo 2. The number of substitution is not enough
# todo 3. train the distance predicate and logic program together 
#todo 4. Using the cluster to get the semantics and build the semantics graph to check the accuracy of rules. ✔️
# todo 5. Show the clustering accuracy and debug the accuracy 