import datetime
import pandas as pd
import os 
import torch 
import itertools
import torch.nn as nn
import argparse
from accelerate import Accelerator
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel, BertTokenizer, BertForMaskedLM
import json
from torch.utils.tensorboard import SummaryWriter
import tqdm
from sklearn.metrics import confusion_matrix
from logic_back.dforl_torch import DeepRuleLayer_v2
'''
This version implement the bert embeddings and feedforward neural network for the propositionalization layer
'''

class NeuralKG(nn.Module):
    # input entities and relations, outputs the occurrence of the triples 
    def __init__(self, embedding_length):
        super(NeuralKG, self).__init__()
        self.checker = nn.Sequential(
                    nn.Linear(embedding_length, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
    def forward(self, x):
        return self.checker(x)
        
        

class NeuralPredicate(nn.Module):
    def __init__(self, embedding_length, number_atoms):
        super(NeuralPredicate, self).__init__()
        self.single_neural_predicate = nn.Sequential(
                    nn.Linear(2 * embedding_length, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
        self.neural_predicates =  nn.ModuleList([self.single_neural_predicate for _ in range(number_atoms)])
    
    def forward(self, x):
        output = []
        for i ,l in enumerate(self.neural_predicates):
            output.append(l(x))
        return output    



class BaseRule():
    def __init__(self):
        pass
        
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
          
    

        

class ParallelDataset(Dataset):
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
    def __init__(self, train_path, valid_path, test_path, epoch=10, batch_size =512, early_stop = 5, fine_tune = False, append_path='', current_dir = '', task_name='', data_info = '', device = 'cuda:1', datetime_model_finetune = None, tokenize_length=10, lr = 5e-5, margin = 1000):
        super(Distance_predicate, self).__init__()
        # import bert and token 
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.early_stop = EarlyStopper(patience=early_stop, min_delta=0.1)
        self.current_dir = current_dir
        self.task_name = task_name
        self.data_info = data_info
        self.tokenize_length = tokenize_length
        self.lr = lr
        self.margin = margin
        # self.make_data_cls_embeddings()
        # self.process_example_cls_embeddings()
        # self.tokenize()
        self.device = device
        self.writer = SummaryWriter(f'{self.current_dir}/cache/{self.task_name}/runs/')
        self.fine_tune = fine_tune
        if datetime_model_finetune == None:
            self.datetime = datetime.datetime.now().strftime("%d%H")
        else: 
            self.datetime = datetime_model_finetune
        if not os.path.exists(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}"):
            os.makedirs(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}")

        self.output_model_path_basic = f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/bert_fine_tune_{data_info}"
        self.output_model_path = self.output_model_path_basic + append_path + '.pt'
        

    
    
    def process_example_cls_embeddings(self):
        
        
        # load the data 
        train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/train_{self.data_info}_cls_embeddings.csv")['train']
        valid_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/valid_{self.data_info}_cls_embeddings.csv")['train']
        test_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/test_{self.data_info}_cls_embeddings.csv")['train']
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenize_function(examples, main):
            token =  tokenizer(examples[main], return_tensors="pt", padding='max_length', truncation=True, max_length=self.tokenize_length)
            token = {f'input_ids_{main}': token['input_ids'], f'attention_mask_{main}': token['attention_mask']}
            return token
        
        for ds, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            print('transfer', ds)
            token_e = data.map(lambda batch: tokenize_function(batch, 'textE'), batched=True)
            token_e.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/{ds}_{self.data_info}_cls_embeddings_token_E_{self.tokenize_length}.hf")
            
            token_r = data.map(lambda batch: tokenize_function(batch, 'textR'), batched=True)
            token_r.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/{ds}_{self.data_info}_cls_embeddings_token_R_{self.tokenize_length}.hf")
            
            token_l = data.map(lambda batch: tokenize_function(batch, 'label'), batched=True)
            token_l.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/{ds}_{self.data_info}_cls_embeddings_token_L_{self.tokenize_length}.hf")
            
            
    
    def embedding_find_fine_tune(self):
        # load train data
        train_data_E = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/train_{self.data_info}_cls_embeddings_token_E_{self.tokenize_length}.hf")
        train_data_R = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/train_{self.data_info}_cls_embeddings_token_R_{self.tokenize_length}.hf")
        train_data_L = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/train_{self.data_info}_cls_embeddings_token_L_{self.tokenize_length}.hf")
        
        valid_data_E  = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/valid_{self.data_info}_cls_embeddings_token_E_{self.tokenize_length}.hf")
        valid_data_R  = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/valid_{self.data_info}_cls_embeddings_token_R_{self.tokenize_length}.hf")
        valid_data_L  = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/valid_{self.data_info}_cls_embeddings_token_L_{self.tokenize_length}.hf")
        
        test_data_E = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/test_{self.data_info}_cls_embeddings_token_E_{self.tokenize_length}.hf")
        test_data_R = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/test_{self.data_info}_cls_embeddings_token_R_{self.tokenize_length}.hf")
        test_data_L = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/test_{self.data_info}_cls_embeddings_token_L_{self.tokenize_length}.hf")

        # # only choose top 5 instance in all data 
        train_data_E = train_data_E.select(range(10))
        train_data_R = train_data_R.select(range(10))
        train_data_L = train_data_L.select(range(10))
        valid_data_E = valid_data_E.select(range(5))
        valid_data_R = valid_data_R.select(range(5))
        valid_data_L = valid_data_L.select(range(5))
        test_data_E = test_data_E.select(range(5))
        test_data_R = test_data_R.select(range(5))
        test_data_L = test_data_L.select(range(5))
        
        # turn them to tensor format 
        train_data_E.set_format('torch', columns=["input_ids_textE", "attention_mask_textE"])
        train_data_R.set_format('torch', columns=["input_ids_textR", "attention_mask_textR"])
        train_data_L.set_format('torch', columns=["input_ids_label", "attention_mask_label"])
        valid_data_E.set_format('torch', columns=["input_ids_textE", "attention_mask_textE"])
        valid_data_R.set_format('torch', columns=["input_ids_textR", "attention_mask_textR"])
        valid_data_L.set_format('torch', columns=["input_ids_label", "attention_mask_label"])
        test_data_E.set_format('torch', columns=["input_ids_textE", "attention_mask_textE"])
        test_data_R.set_format('torch', columns=["input_ids_textR", "attention_mask_textR"])
        test_data_L.set_format('torch', columns=["input_ids_label", "attention_mask_label"])
        
        

        # make three train_data_E, train_data_R, train_data_L dataset together into a dataloader 
        paralle_data = ParallelDataset(train_data_E, train_data_R, train_data_L)
        train_data_con = torch.utils.data.DataLoader(paralle_data, batch_size=self.batch_size, shuffle=False)
        
        parallel_valid = ParallelDataset(valid_data_E, valid_data_R, valid_data_L)
        valid_data_con = torch.utils.data.DataLoader(parallel_valid, batch_size=self.batch_size, shuffle=False)
        
        parallel_test = ParallelDataset(test_data_E, test_data_R, test_data_L)
        test_data_con = torch.utils.data.DataLoader(parallel_test, batch_size=self.batch_size, shuffle=False)
        
        # load the model
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        model.to(self.device)
        # define the optimizer
        optimizer = AdamW(model.parameters(), lr=self.lr)
        
        if self.fine_tune:
            checkpoint = torch.load(self.output_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        
        #MSE as loss 
        criterion_mse = nn.L1Loss()
        # todo: check 
        criterion = nn.MarginRankingLoss(margin=self.margin, reduction='mean')
        # criterion = nn.CrossEntropyLoss()
        # set data to torch tensor
        model.train()
        for single_epoch in range(self.epoch):
            print('Epoch:', single_epoch)
            # obtain the first embedding 
            epoch_loss = 0
            for batch in tqdm.tqdm(train_data_con):
                e_input_id = batch["id_e"].to(self.device)
                e_attention_mask = batch["mask_e"].to(self.device)
                r_input_id = batch["id_r"].to(self.device)
                r_attention_mask = batch["mask_r"].to(self.device)
                l_input_id = batch["id_l"].to(self.device)
                l_attention_mask = batch["mask_l"].to(self.device)
                
                neg_e_input_id = batch["neg_head_id"].to(self.device)
                neg_e_attention_mask = batch["neg_head_mask"].to(self.device)
                neg_l_input_id = batch["neg_tail_id"].to(self.device)
                neg_l_attention_mask = batch["neg_tail_mask"].to(self.device)
            
                
                
                e_output = model(input_ids=e_input_id, attention_mask=e_attention_mask, output_hidden_states=True)
                r_output = model(input_ids=r_input_id, attention_mask=r_attention_mask, output_hidden_states=True)
                l_output = model(input_ids=l_input_id, attention_mask=l_attention_mask, output_hidden_states=True)
                # embedd the cls token
                e_hidden_states = self.get_hidden_states(e_output, token_ids_word=0)
                r_hidden_states = self.get_hidden_states(r_output, token_ids_word=0)
                l_hidden_states = self.get_hidden_states(l_output, token_ids_word=0)
                
                # get negative output 
                neg_e_output = model(input_ids=neg_e_input_id, attention_mask=neg_e_attention_mask, output_hidden_states=True)
                neg_l_output = model(input_ids=neg_l_input_id, attention_mask=neg_l_attention_mask, output_hidden_states=True)
                neg_e_embeddings = self.get_hidden_states(neg_e_output, token_ids_word=0)
                neg_l_embeddings = self.get_hidden_states(neg_l_output, token_ids_word=0)
                
                # obatin the distance 
                distance = self.get_distance(e_hidden_states, r_hidden_states, l_hidden_states)
                
                negative_distance = self.get_distance(neg_e_embeddings, r_hidden_states, neg_l_embeddings)
                
                # calculate the loss
                optimizer.zero_grad()
                # loss_mse = criterion(distance, torch.zeros_like(distance))
                loss = criterion(distance, negative_distance, -1*torch.ones_like(distance, device=self.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / len(train_data_con)
            print(f"[Train] Epoch: {single_epoch}, Loss: {epoch_loss}")
            self.writer.add_scalar("Loss/train", epoch_loss, single_epoch)
            # eval the model 
            model.eval()
            epoch_loss_valid = 0
            for batch in tqdm.tqdm(valid_data_con):
                e_input_id = batch["id_e"].to(self.device)
                e_attention_mask = batch["mask_e"].to(self.device)
                r_input_id = batch["id_r"].to(self.device)
                r_attention_mask = batch["mask_r"].to(self.device)
                l_input_id = batch["id_l"].to(self.device)
                l_attention_mask = batch["mask_l"].to(self.device)
                e_output = model(input_ids=e_input_id, attention_mask=e_attention_mask, output_hidden_states=True)
                r_output = model(input_ids=r_input_id, attention_mask=r_attention_mask, output_hidden_states=True)
                l_output = model(input_ids=l_input_id, attention_mask=l_attention_mask, output_hidden_states=True)
                # embedd the cls token
                e_hidden_states = self.get_hidden_states(e_output, token_ids_word=0)
                r_hidden_states = self.get_hidden_states(r_output, token_ids_word=0)
                l_hidden_states = self.get_hidden_states(l_output, token_ids_word=0)
                # obatin the distance 
                distance = self.get_distance(e_hidden_states, r_hidden_states, l_hidden_states)
                # calculate the loss
                loss = criterion_mse(distance, torch.zeros_like(distance))
                epoch_loss_valid += loss.item()
            epoch_loss_valid = epoch_loss_valid / len(valid_data_con)
            print(f"[Valid] Epoch: {single_epoch}, Loss: {epoch_loss_valid}")
            self.writer.add_scalar("Loss/valid", epoch_loss_valid, single_epoch)
            self.writer.flush()
            if self.early_stop.early_stop(epoch_loss_valid):
                print("[Early Stopping Training]")
                break
            self.save(model, optimizer, append_path=f"_{epoch_loss}")
            
            
        # test the model
        model.eval()
        epoch_loss_test = 0
        for batch in tqdm.tqdm(test_data_con):
            e_input_id = batch["id_e"].to(self.device)
            e_attention_mask = batch["mask_e"].to(self.device)
            r_input_id = batch["id_r"].to(self.device)
            r_attention_mask = batch["mask_r"].to(self.device)
            l_input_id = batch["id_l"].to(self.device)
            l_attention_mask = batch["mask_l"].to(self.device)
            e_output = model(input_ids=e_input_id, attention_mask=e_attention_mask, output_hidden_states=True)
            r_output = model(input_ids=r_input_id, attention_mask=r_attention_mask, output_hidden_states=True)
            l_output = model(input_ids=l_input_id, attention_mask=l_attention_mask, output_hidden_states=True)
            # embedd the cls token
            e_hidden_states = self.get_hidden_states(e_output, token_ids_word=0)
            r_hidden_states = self.get_hidden_states(r_output, token_ids_word=0)
            l_hidden_states = self.get_hidden_states(l_output, token_ids_word=0)
            # obatin the distance 
            distance = self.get_distance(e_hidden_states, r_hidden_states, l_hidden_states)
            # calculate the loss
            loss = criterion_mse(distance, torch.zeros_like(distance))
            epoch_loss_test += loss.item()
        epoch_loss_test = epoch_loss_test / len(test_data_con)
        print(f"[TEST] Loss: {epoch_loss_test}")
        
                
        # save the model
        self.save(model, optimizer)
        

    # inference
    def inference_distance(self, model_path = ''):
        infere_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/inference_{self.data_info}_cls_embeddings.csv")['train']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        def tokenize_function(examples, main):
            token =  tokenizer(examples[main], return_tensors="pt", padding='max_length', truncation=True, max_length=self.tokenize_length)
            token = {f'input_ids_{main}': token['input_ids'], f'attention_mask_{main}': token['attention_mask']}
            return token

        test_data_E = infere_data.map(lambda batch: tokenize_function(batch, 'textE'), batched=True)
        test_data_R = infere_data.map(lambda batch: tokenize_function(batch, 'textR'), batched=True)
        test_data_L = infere_data.map(lambda batch: tokenize_function(batch, 'label'), batched=True)

        # load the local data 
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)


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
            e_output = model(input_ids=e_input_id, attention_mask=e_attention_mask, output_hidden_states=True)
            r_output = model(input_ids=r_input_id, attention_mask=r_attention_mask, output_hidden_states=True)
            l_output = model(input_ids=l_input_id, attention_mask=l_attention_mask, output_hidden_states=True)
            # embedd the cls token
            e_hidden_states = self.get_hidden_states(e_output, token_ids_word=0)
            r_hidden_states = self.get_hidden_states(r_output, token_ids_word=0)
            l_hidden_states = self.get_hidden_states(l_output, token_ids_word=0)
            # obatin the distance 
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
            epoch_loss_test += loss.item()
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
            
    def tokenize(self):
        # load the data 
        train_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/train_{self.data_info}_mask.csv")['train']
        valid_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/valid_{self.data_info}_mask.csv")['train']
        test_data = load_dataset('csv', data_files=f"{self.current_dir}/cache/{self.task_name}/test_{self.data_info}_mask.csv")['train']
        
        # load the model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                # Tokenize input
        
        def tokenize_function(examples):
            token =  tokenizer(examples["text"], return_tensors="pt", padding='max_length', truncation=True, max_length=100)
            return token
        
        for ds, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            print('transfer', ds)
            masked_data  = data.map(tokenize_function, batched=True)
            masked_data.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/{ds}_{self.data_info}_mask_token.hf")
        
        
        
    def finetune(self):
        
        train_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/train_{self.data_info}_mask_token.hf")
        valid_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/valid_{self.data_info}_mask_token.hf")
        test_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/test_{self.data_info}_mask_token.hf")
        
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        model.to(self.device)
        
        # set data to torch tensor 
        train_data.set_format('torch', columns=["input_ids", "attention_mask","label"])
        
        model.train()
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        for epoch in range(3):
            total_loss = 0
            for batch in tqdm.tqdm(train_dataloader):
                model.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model()
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
    
    def inference(self, text, top_k=5):
        # inference the distance between two entities 
        # Tokenize input
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model.eval()
        text = "[CLS] %s [SEP]"%text
        tokenized_text = tokenizer.tokenize(text)
        masked_index = tokenized_text.index("[MASK]")
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

        for i, pred_idx in enumerate(top_k_indices):
            predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
            token_weight = top_k_weights[i]
            print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
        
        

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
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



class DifferentiablePropositionalization(nn.Module, BaseRule):
    def __init__(self, entity_path, relation_path, epoch=10, batch_size =512, data_info = '', early_stop = 5):
        super(DifferentiablePropositionalization, self).__init__()
        # import bert and token 
        self.entity_path = entity_path
        self.relation_path = relation_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.early_stop = EarlyStopper(patience=early_stop, min_delta=0.1)
        self.number_variable = 3
        self.target_relation = 'Make_statement'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(f'{self.current_dir}/cache/{self.task_name}/runs/')
        self.datetime = datetime.datetime.now().strftime("%d%H")
        if not os.path.exists(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}"):
            os.makedirs(f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}")
        self.output_model_path_basic = f"{self.current_dir}/cache/{self.task_name}/out/{self.datetime}/soft_proposition_{data_info}"
        self.variable_perturbations = list(itertools.permutations(range(self.number_variable), 2))
        self.get_index_X_Y = self.variable_perturbations.index((0,1))
        # self.variable_perturbations.remove((0,1))
        self.number_variable_terms  = len(self.variable_perturbations)
        self.variable_perturbations = torch.tensor(self.variable_perturbations)
        accelerator = Accelerator()
        # write the pertumation to the file
        with open(f"{self.current_dir}/cache/{self.task_name}/variable_perturbations.txt", "w") as f:
            f.write(str(self.variable_perturbations))
            f.close()

    def make_tokeniz(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # load the data
        entity_data = load_dataset('csv', data_files=self.entity_path)['train']
        relation_data = load_dataset('csv', data_files=self.relation_path)['train']
        
        def tokenize_function(examples):
            token =  tokenizer(examples["text"], return_tensors="pt", padding='max_length', truncation=True, max_length=100)
            return token
        
        entity_data = entity_data.map(tokenize_function, batched=True)
        relation_data = relation_data.map(tokenize_function, batched=True)
        # save them to disk
        entity_data.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/entity_token.hf")
        relation_data.save_to_disk(f"{self.current_dir}/cache/{self.task_name}/relation_token.hf")
    
    def soft_pro(self):
        '''
        Read all entities and process and propositionalalization and rule learning task together 
        '''
        # read all entities 
        entity_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/entity_token.hf")
        relation_data = load_from_disk(f"{self.current_dir}/cache/{self.task_name}/relation_token.hf")
        self.tar_predicate_index = None
        number_relations = len(relation_data)
        for item in range(number_relations):
            if relation_data[item]['text'] == self.target_relation.replace('_', ' '):
                self.tar_predicate_index = item
                break
        self.target_atom_index = number_relations * self.get_index_X_Y + self.tar_predicate_index
        self.target_atom_index = self.tar_predicate_index * self.number_variable_terms + self.get_index_X_Y
        entity_data.set_format('torch', columns=["input_ids", "attention_mask"])
        relation_data.set_format('torch', columns=["input_ids", "attention_mask"])





        relation_data = DataLoader(relation_data, batch_size=len(relation_data), shuffle=False)

        
        self.relation_index = torch.tensor(range(len(relation_data)))
        self.relation_index = self.relation_index.unsqueeze(-1)
        self.relation_index = self.relation_index.expand(self.number_variable_terms, -1, -1)
        # load the data and neural predicate model
        substation_data = SubData(entity_data, self.number_variable)
        substation_data = torch.utils.data.DataLoader(substation_data, batch_size=1, shuffle=False)

        predicate = BertForMaskedLM.from_pretrained('bert-base-uncased')
        #! load pretrain weights 
        # predicate.load_state_dict(torch.load(f"{self.current_dir}/cache/{self.task_name}/out/0106/bert_fine_tune_balance_0.0.pt")['model_state_dict'])
        predicate.to(self.device)
        
        # Load neural logic rule learning model 
        logic_model = DeepRuleLayer_v2(number_relations*self.number_variable_terms, rule_number = 2, default_alpha = 10,device = self.device, output_rule_file=f"{self.current_dir}/cache/{self.task_name}/rule_output.txt", t_relation=f'{self.task_name}', task=f'{self.task_name}', t_arity=2,time_stamp='' )
        logic_model.to(self.device)
        
        # define optimizer 
        logic_optimizer = torch.optim.Adam(logic_model.parameters(), lr=0.001)
        best_acc_train  = 0 
        predicate, logic_model = self.accelerator.prepare([predicate, logic_model], device=self.device)
        
        for batch_index, single_bath in enumerate(relation_data):
            if batch_index > 0:
                raise ValueError("Only one batch is allowed")
            single_bath = {k: v.to(self.device) for k, v in single_bath.items()}
            all_r_embeddings = predicate(input_ids=single_bath['input_ids'], attention_mask=single_bath['attention_mask'], output_hidden_states=True)
            all_r_embeddings = self.get_hidden_states(all_r_embeddings, token_ids_word=0)
        
        # start training
        logic_model.train()
        for single_epoch in range(self.epoch):
            for batch_index in tqdm.tqdm(substation_data):
                # get the embedding of the entity 
                batch, index = batch_index
                index = index
                index = index.squeeze(0)
                first_entity_id = batch['first_entity_id'].to(self.device)  # obj entity 
                first_entity_mask = batch['first_entity_mask'].to(self.device)
                second_entity_id = batch['second_entity_id'].to(self.device) # label entity / sub entity / second entity 
                second_entity_mask = batch['second_entity_mask'].to(self.device)
                third_entity_id = batch['third_entity_id'].to(self.device)
                third_entity_mask = batch['third_entity_id'].to(self.device)
                
                # get the distance between the entity and relation
                first_output = predicate(input_ids=first_entity_id, attention_mask=first_entity_mask, output_hidden_states=True)
                second_output = predicate(input_ids=second_entity_id, attention_mask=second_entity_mask, output_hidden_states=True)
                third_output = predicate(input_ids=third_entity_id, attention_mask=third_entity_mask, output_hidden_states=True)
                # embedd the cls token
                first_embeddings = self.get_hidden_states(first_output, token_ids_word=0)
                second_embeddings = self.get_hidden_states(second_output, token_ids_word=0)
                third_embeddings = self.get_hidden_states(third_output, token_ids_word=0)
            
                
                # obtain the substitution for all atoms in body 
                # get the all variable perturbation and return the substitutions based on X, Y, and Z variable
                perturbations = index[self.variable_perturbations]
                # first list 
                perturbations = perturbations.unsqueeze(-1).expand(-1,-1,768)
                first_index = index[0].unsqueeze(-1).expand(768)
                perturbations = torch.where(perturbations == first_index, first_embeddings, perturbations )
                perturbations = torch.where(perturbations == index[1].unsqueeze(-1).expand(768), second_embeddings, perturbations )
                perturbations = torch.where(perturbations == index[2].unsqueeze(-1).expand(768), third_embeddings, perturbations )
                # triples_embeddgins = torch.where(perturbations == index[0], first_embeddings, (torch.where(perturbations == index[1], second_embeddings, (torch.where(perturbations == index[2], third_embeddings, 1)))))
                # combine the entity and relation
                perturbations = perturbations.unsqueeze(1)
                expanded_per = perturbations.expand(-1, number_relations, -1, -1)
                all_r_embeddings = all_r_embeddings.unsqueeze(0)
                all_r_embeddings  = all_r_embeddings.expand(self.number_variable_terms, -1, -1)
                all_r_embeddings = all_r_embeddings.unsqueeze(2)
                # include the relations 
                triples = torch.cat([expanded_per, all_r_embeddings], dim=2)        
                triples = triples.reshape(-1, 3, 768) # R_0(X,Y), R_1(X,Y), ...,R_n(X,Y) ,R_0(Y,X),.., R_n(Y,X), ... X (e1, e2, r) X embedding length 
                # replace elements to embeddings )
                
                
                
                
                # get the distance between the entity and relation
                e_hidden_states = triples[:,0,:]
                r_hidden_states = triples[:,2,:]
                l_hidden_states = triples[:,1,:]
                distance = self.get_distance(e_hidden_states, r_hidden_states, l_hidden_states)
                body_predicate_labels_predicated = torch.where(distance - 1000 < 0, torch.ones_like(distance), torch.zeros_like(distance))
                head_predicate_labels_predicated = body_predicate_labels_predicated[self.target_atom_index]
                body_predicate_labels_predicated = torch.cat([body_predicate_labels_predicated[:self.target_atom_index], body_predicate_labels_predicated[self.target_atom_index+1:]])
                # find the target order of X and Y 
                
                # append rule learning module 
                predicated = logic_model(body_predicate_labels_predicated)
                loss = nn.MSELoss()(predicated, head_predicate_labels_predicated)
                logic_optimizer.zero_grad()
                # loss.backward()
                self.accelerator.backward(loss)
                logic_optimizer.step()
                # print the loss
                print(f"Epoch: {single_epoch}, Loss: {loss.item()}")
                self.writer.add_scalar("Loss/train", loss.item(), single_epoch)
                self.writer.flush()
                if self.early_stop.early_stop(loss.item()):
                    print("[Early Stopping Training]")
                    break
                self.save(logic_model, logic_optimizer, append_path=f"_{loss.item()}")
                


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
    def __init__(self, entity, number_variable):
        self.entity = entity
        self.number_variable = number_variable
    def __len__(self):
        return len(self.entity) * (len(self.entity)-1) * (len(self.entity)-2)
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






def soft_pro_train_rules():
# if __name__ == "__main__":
    current_dir = os.getcwd()
    folder_name = '/gammaILP/'
    current_dir = current_dir + folder_name
    task_name = 'icews14'
    data_info = 'balance'
    entity_path = f"{current_dir}/cache/{task_name}/all_entities_sampled.csv" # all entities from train file
    relation_path = f"{current_dir}/cache/{task_name}/all_relations_sampled.csv" # all relations from train file
    torch.manual_seed(0)
    model = DifferentiablePropositionalization(entity_path, relation_path, epoch=30, batch_size =2)
    # model.make_tokeniz()
    model.soft_pro()



def pre_train_predicate_neural(args):
    current_dir = args.folder_name
    task_name = args.task
    data_info = args.data_info
    train_path = f"{current_dir}/cache/{task_name}/triple_train_{data_info}.csv"
    valid_path = f"{current_dir}/cache/{task_name}/triple_train_{data_info}.csv"
    test_path = f"{current_dir}/cache/{task_name}/triple_train_{data_info}.csv"
    
    model = Distance_predicate(train_path, valid_path, test_path, epoch=args.epoch, batch_size =args.batch_size, data_info=data_info, fine_tune = args.fine_tune, datetime_model_finetune=args.model_path,current_dir=current_dir, task_name=task_name, early_stop=args.early_stop, device=args.device, tokenize_length = args.tokenize_length,lr=args.lr, margin=args.margin)
    if args.inf == True:
        model.inference_distance(model_path = f'{args.model_path_parent}/{args.model_path}/bert_fine_tune_balance.pt')
    else:
        model.embedding_find_fine_tune()
    # model.inference_distance(model_path = f'{args.model_path_parent}/{args.model_path}/bert_fine_tune_balance.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='icews14')
    parser.add_argument("--data_info", type=str, default='balance')
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--fine_tune", type=bool, default=False)
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--folder-name", type=str, default=f'{os.getcwd()}/gammaILP/')
    parser.add_argument('--model-path-parent', type=str, default=f'{os.getcwd()}/gammaILP/cache/icews14/out/')
    parser.add_argument('--model-path', type=str, default='test_demo')
    parser.add_argument('--tokenize_length', type=int, default=10)
    parser.add_argument('--lr', type=int, default=5e-5)
    parser.add_argument('--inf', action='store_true')
    parser.add_argument('--margin', type=int, default=1000)
    args = parser.parse_args()
    # record the parameters 
    print(args)
    torch.manual_seed(0)
    pre_train_predicate_neural(args)
    # soft_pro_train_rules()

