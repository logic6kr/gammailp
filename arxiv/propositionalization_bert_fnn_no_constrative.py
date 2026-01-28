import datetime
import pandas as pd
import os 
import torch 
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel, BertTokenizer, BertForMaskedLM
import json
from torch.utils.tensorboard import SummaryWriter
import tqdm
from sklearn.metrics import confusion_matrix
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
        random_head = torch.randint(0, 2, (1,))
        random_index = torch.randint(0, self.min_length, (1,))
        if random_index == idx:
            random_index = (random_index + 1) % self.min_length
        if random_head == 1:
            neg_head = self.e[random_index]
            neg_tail = self.l[idx]
        else:
            neg_head = self.e[idx]
            neg_tail = self.l[random_index]

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
        
        
class Distance_predicate(nn.Module):
    def __init__(self, train_path, valid_path, test_path, epoch=10, batch_size =512, data_info = '', early_stop = 5, fine_tune = False, append_path=''):
        super(Distance_predicate, self).__init__()
        # import bert and token 
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.early_stop = EarlyStopper(patience=early_stop, min_delta=0.1)

        # self.make_data_cls_embeddings()
        # self.process_example_cls_embeddings()
        # self.tokenize()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(f'{current_dir}/cache/{task_name}/runs/')
        self.fine_tune = fine_tune
        self.output_model_path_basic = f"{current_dir}/cache/{task_name}/out/bert_fine_tune_{data_info}"
        self.output_model_path = self.output_model_path_basic + append_path + '.pt'

    # def make_data_mask_group_sentence(self):
    #     '''
    #     This code add mask token in a group sentences
    #     '''
    #     # load the dataset 
    #     with open(self.train_path, "r") as f:
    #         train_data = pd.read_csv(f)
    #         f.close()
    #     with open(self.valid_path, "r") as f:
    #         valid_data = pd.read_csv(f)
    #         f.close()
    #     with open(self.test_path, "r") as f:
    #         test_data = pd.read_csv(f)
    #         f.close()
    #     for ds, data_type in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
    #         # make the data
    #         data_type = data_type[data_type['label']==1]
    #         data_type = data_type['triple'].tolist()
    #         train_label = {'text':[], 'label':[]}
    #         print('processing', ds)
    #         for item in tqdm.tqdm(data_type):
    #             # remove " in sentence 
    #             sentence = item.split('$')
    #             first_entity = sentence[0].replace('_', ' ')
    #             relation = sentence[1].replace('_', ' ')
    #             second_entity = sentence[2].replace('_', ' ')
    #             sentence = f"{first_entity} {relation} [MASK]"
    #             sentence = f"{sentence}"
    #             label = second_entity
    #             train_label['text'].append(sentence)
    #             train_label['label'].append(label)
    #         train_label = pd.DataFrame(train_label)
    #         train_label.to_csv(f"{current_dir}/cache/{task_name}/{ds}_{data_info}_mask.csv", index=False)
        
    # def make_data_cls_embeddings(self):
    #     '''
    #     This code add cls token in entity and relation, and use their embeddings as key to find another embeddings 
    #     '''
    #     # load the dataset
    #     with open(self.train_path, "r") as f:
    #         train_data = pd.read_csv(f)
    #         f.close()
    #     with open(self.valid_path, "r") as f:
    #         valid_data = pd.read_csv(f)
    #         f.close()
    #     with open(self.test_path, "r") as f:
    #         test_data = pd.read_csv(f)
    #         f.close()
    #     for ds, data_type in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
    #         # make the data
    #         data_type = data_type[data_type['label']==1]
    #         data_type = data_type['triple'].tolist()
    #         train_label = {'textE':[], 'textR':[], 'label':[]}
    #         print('processing', ds)
    #         for item in tqdm.tqdm(data_type):
    #             # remove " in sentence 
    #             sentence = item.split('$')
    #             first_entity = sentence[0].replace('_', ' ')
    #             relation = sentence[1].replace('_', ' ')
    #             second_entity = sentence[2].replace('_', ' ')
    #             sentenceE = f"{first_entity}"
    #             sentenceR  = f"{relation}"
    #             label = f"{second_entity}"
    #             train_label['textE'].append(sentenceE)
    #             train_label['textR'].append(sentenceR)
    #             train_label['label'].append(label)
    #         train_label = pd.DataFrame(train_label)
    #         train_label.to_csv(f"{current_dir}/cache/{task_name}/{ds}_{data_info}_cls_embeddings.csv", index=False)

    def process_example_cls_embeddings(example):
        
        # load the data 
        train_data = load_dataset('csv', data_files=f"{current_dir}/cache/{task_name}/train_{data_info}_cls_embeddings.csv")['train']
        valid_data = load_dataset('csv', data_files=f"{current_dir}/cache/{task_name}/valid_{data_info}_cls_embeddings.csv")['train']
        test_data = load_dataset('csv', data_files=f"{current_dir}/cache/{task_name}/test_{data_info}_cls_embeddings.csv")['train']
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenize_function(examples, main):
            token =  tokenizer(examples[main], return_tensors="pt", padding='max_length', truncation=True, max_length=100)
            token = {f'input_ids_{main}': token['input_ids'], f'attention_mask_{main}': token['attention_mask']}
            return token


        
        for ds, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            print('transfer', ds)
            token_e = data.map(lambda batch: tokenize_function(batch, 'textE'), batched=True)
            token_e.save_to_disk(f"{current_dir}/cache/{task_name}/{ds}_{data_info}_cls_embeddings_token_E.hf")
            
            token_r = data.map(lambda batch: tokenize_function(batch, 'textR'), batched=True)
            token_r.save_to_disk(f"{current_dir}/cache/{task_name}/{ds}_{data_info}_cls_embeddings_token_R.hf")
            
            token_l = data.map(lambda batch: tokenize_function(batch, 'label'), batched=True)
            token_l.save_to_disk(f"{current_dir}/cache/{task_name}/{ds}_{data_info}_cls_embeddings_token_L.hf")
            
            
    

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
        # get the distance between two entities
        # d(h + ℓ,t) = |h|_2^2 + |l|_2^2 + |t|_2^2 −2 hT t + ℓT (t−h)
        e_norm = torch.linalg.norm(e, ord=2, dim=1)
        r_norm = torch.linalg.norm(r, ord=2, dim=1)
        l_norm = torch.linalg.norm(l, ord=2, dim=1)
        # distance = e_norm**2 + r_norm**2 + l_norm**2 - 2 * (torch.dot(e, l) + torch.dot(r, l - e))
        distance = e_norm**2 + r_norm**2 + l_norm**2 - 2 * (torch.diagonal(torch.mm(e, torch.transpose(l,0,1)),0) + torch.diagonal(torch.mm(r, torch.transpose(l - e,0,1))))
        return distance
            
    def embedding_find_fine_tune(self):
        # load train data
        train_data_E = load_from_disk(f"{current_dir}/cache/{task_name}/train_{data_info}_cls_embeddings_token_E.hf")
        train_data_R = load_from_disk(f"{current_dir}/cache/{task_name}/train_{data_info}_cls_embeddings_token_R.hf")
        train_data_L = load_from_disk(f"{current_dir}/cache/{task_name}/train_{data_info}_cls_embeddings_token_L.hf")
        
        valid_data_E  = load_from_disk(f"{current_dir}/cache/{task_name}/valid_{data_info}_cls_embeddings_token_E.hf")
        valid_data_R  = load_from_disk(f"{current_dir}/cache/{task_name}/valid_{data_info}_cls_embeddings_token_R.hf")
        valid_data_L  = load_from_disk(f"{current_dir}/cache/{task_name}/valid_{data_info}_cls_embeddings_token_L.hf")
        
        test_data_E = load_from_disk(f"{current_dir}/cache/{task_name}/test_{data_info}_cls_embeddings_token_E.hf")
        test_data_R = load_from_disk(f"{current_dir}/cache/{task_name}/test_{data_info}_cls_embeddings_token_R.hf")
        test_data_L = load_from_disk(f"{current_dir}/cache/{task_name}/test_{data_info}_cls_embeddings_token_L.hf")

        # # only choose top 5 instance in all data 
        # train_data_E = train_data_E.select(range(5))
        # train_data_R = train_data_R.select(range(5))
        # train_data_L = train_data_L.select(range(5))
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
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        if self.fine_tune:
            checkpoint = torch.load(self.output_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        
        #MSE as loss 
        # criterion = nn.L1Loss()
        # todo: check 
        criterion = nn.MarginRankingLoss(margin=1.0, reduction='none')
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
                
                # # prepare the negative examples 
                # head_or_tail = torch.randint(0, 2, e_input_id.size()[0], device=self.device)
                # head_or_tail = head_or_tail.unsqueeze(1)
                # random_entities_index = torch.randint(0, len(train_data_E), e_input_id.size()[0], device=self.device)
                # broken_heads_id = torch.where(head_or_tail == 1, train_data_E[random_entities_index]["input_ids_label"] , e_input_id)
                # broken_tails_id = torch.where(head_or_tail == 0, train_data_E[random_entities_index]["input_ids_label"] , l_input_id)
                
                
                
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
                
                # obatin the distance 
                distance = self.get_distance(e_hidden_states, r_hidden_states, l_hidden_states)
                
                negative_distance = self.get_distance(neg_e_output, r_hidden_states, neg_l_output)
                
                # calculate the loss
                optimizer.zero_grad()
                # loss = criterion(distance, torch.zeros_like(distance))
                loss = criterion(distance, negative_distance, torch.tensor([-1], device=self.device))
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
                loss = criterion(distance, torch.zeros_like(distance))
                epoch_loss_valid += loss.item()
            epoch_loss_valid = epoch_loss_valid / len(valid_data_con)
            print(f"[Valid] Epoch: {single_epoch}, Loss: {epoch_loss_valid}")
            self.writer.add_scalar("Loss/valid", epoch_loss_valid, single_epoch)
            self.writer.flush()
            if self.early_stop.early_stop(epoch_loss_valid):
                print("[Early Stopping Training]")
                break
            # self.save(model, optimizer, append_path=f"_{epoch_loss}")
            
            
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
            loss = criterion(distance, torch.zeros_like(distance))
            epoch_loss_test += loss.item()
        epoch_loss_test = epoch_loss_test / len(test_data_con)
        print(f"[TEST] Loss: {epoch_loss_test}")
        
                
        # save the model
        self.save(model, optimizer)
        

    # inference
    def inference_distance(self):
        infere_data = load_dataset('csv', data_files=f"{current_dir}/cache/{task_name}/inference_{data_info}_cls_embeddings.csv")['train']
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
        checkpoint = torch.load(self.output_model_path, map_location='cpu')
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
        # save
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, self.output_model_path_basic+append_path+'.pt')
        # check if model number is larger then 4, then delete the oldest file
        model_files = os.listdir(f"{current_dir}/cache/{task_name}/out/")
        if len(model_files) > 4:
            loss_files = []
            try:
                for i in model_files:
                    loss = float(i.split('_')[-1][:-3])    
                    loss_files.append(loss)
                # model_files = [int(i.split('_')[-1].split('.')[0]) for i in model_files]
            except:
                pass
            loss_files.sort()
            os.remove(f"{current_dir}/cache/{task_name}/out/bert_fine_tune_{data_info}_{loss_files[0]}.pt")
            

            
    def tokenize(self):
        # load the data 
        train_data = load_dataset('csv', data_files=f"{current_dir}/cache/{task_name}/train_{data_info}_mask.csv")['train']
        valid_data = load_dataset('csv', data_files=f"{current_dir}/cache/{task_name}/valid_{data_info}_mask.csv")['train']
        test_data = load_dataset('csv', data_files=f"{current_dir}/cache/{task_name}/test_{data_info}_mask.csv")['train']
        
        # load the model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                # Tokenize input
        
        def tokenize_function(examples):
            token =  tokenizer(examples["text"], return_tensors="pt", padding='max_length', truncation=True, max_length=100)
            return token
        
        for ds, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            print('transfer', ds)
            masked_data  = data.map(tokenize_function, batched=True)
            masked_data.save_to_disk(f"{current_dir}/cache/{task_name}/{ds}_{data_info}_mask_token.hf")
        
        
        
    def finetune(self):
        
        train_data = load_from_disk(f"{current_dir}/cache/{task_name}/train_{data_info}_mask_token.hf")
        valid_data = load_from_disk(f"{current_dir}/cache/{task_name}/valid_{data_info}_mask_token.hf")
        test_data = load_from_disk(f"{current_dir}/cache/{task_name}/test_{data_info}_mask_token.hf")
        
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
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# # differentiable propositionalization layer
# class DifferentiablePropositionalization(nn.Module):
#     def __init__(self, train_path, valid_path, test_path, epoch=10, batch_size =512, data_info = '', early_stop = 5):
#         super(DifferentiablePropositionalization, self).__init__()
#         # make embeddings for input entities 
#         self.epoch = epoch
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.bath_size = batch_size
#         self.embedding_length = BertConfig().from_pretrained("bert-base-cased").hidden_size
#         self.data_path = f"{current_dir}/cache/{task_name}"
#         self.data_info = data_info
#         self.early_stop = EarlyStopper(patience=early_stop, min_delta=0.1)
#         # load json file
#         with open(f"{self.data_path}/config.jsonl", "r") as f:
#             self.config = json.load(f)
#             f.close()
#         if not os.path.exists(f"{self.data_path}/train_{self.data_info}.hf"):
#             self.make_embeddings_data(train_path, valid_path, test_path)
#         # set the number of atoms 
#         self.number_atoms = self.config["Number_atoms"]

#     def make_single_embeddings(self, data: list[str]):
#         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#         self.LM = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        
#         def get_embedding(x: list[str]):
#             language_facts = [i.replace('$', ' ') for i in x]
#             facts = self.tokenizer(language_facts, return_tensors="pt", padding='max_length', truncation=True, max_length=100)
#             with torch.no_grad():
#                 facts = {k: v.to(self.device) for k, v in facts.items()}
#                 outputs = self.LM(**facts)
#             hidden_states = outputs.hidden_states
#             # get the last hidden state
#             hidden_states = hidden_states[-1]
#             # choose CLS token as the embedding of the entity
#             embeddings = hidden_states[:, 0, :]
#             return embeddings

#         # load model and data to GPU 
#         self.LM.to(self.device)
        
#         embeddings = get_embedding(data)
#         return embeddings

        
#     def make_embeddings_data(self, train_path, valid_path, test_path):
#         # load data 
#         # embedding entity+relation+entity with bert 
#         train_dataset = load_dataset('csv', data_files=train_path)['train']
#         valid_dataset = load_dataset('csv', data_files=valid_path)['train']
#         test_dataset = load_dataset('csv', data_files=test_path)['train']
        
                
#         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#         self.LM = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        
#         def get_embedding(x):
#             # first_ent = [i.split('$')[0] for i in x["triple"]]
#             # relations = [i.split('$')[1] for i in x["triple"]]
#             # second_ent = [i.split('$')[2] for i in x["triple"]]
#             language_facts = [i.replace('$', ' ') for i in x["triple"]]
#             # expect tuple the rest would be label list 
#             # labels = [x[i] for i in list(x.keys())[1:]]
#             labels = x["label"]
            

#             facts = self.tokenizer(language_facts, return_tensors="pt", padding='max_length', truncation=True, max_length=100)
#             with torch.no_grad():
#                 facts = {k: v.to(self.device) for k, v in facts.items()}
#                 outputs = self.LM(**facts)
#             hidden_states = outputs.hidden_states
#             # get the last hidden state
#             hidden_states = hidden_states[-1]
#             # choose CLS token as the embedding of the entity
#             embeddings = hidden_states[:, 0, :]
#             new_pairs = {"triple": embeddings, "label": labels}
                
                            
#             # inputs = self.tokenizer(first_ent, return_tensors="pt", padding='max_length', truncation=True, max_length=20)
#             # with torch.no_grad():
#             #     # load to GPU 
#             #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
#             #     outputs = self.LM(**inputs)
#             # hidden_states = outputs.hidden_states
            
#             # inputs_second = self.tokenizer(second_ent, return_tensors="pt", padding='max_length', truncation=True, max_length=20)
#             # with torch.no_grad():
#             #     inputs_second = {k: v.to(self.device) for k, v in inputs_second.items()}
#             #     outputs_second = self.LM(**inputs_second)
#             # hidden_states_second = outputs_second.hidden_states
            
#             # first_ent = hidden_states[-1]
#             # # choose CLS token as the embedding of the entity
#             # mean_first_ent = first_ent[:, 0, :]
#             # # mean_first_ent = first_ent.mean(dim=1)
            
#             # second_ent = hidden_states_second[-1]
#             # mean_second_ent = first_ent[:, 0, :]
#             # # mean_second_ent = second_ent.mean(dim=1)
            
#             # # concat the two embeddings of entities            
#             # embeddings = torch.cat((mean_first_ent, mean_second_ent), dim=1)
#             # labels = torch.tensor(labels).transpose(0, 1)
#             # new_pairs = {"tuple": embeddings, "labels": labels}
#             return new_pairs

#         # load model and data to GPU 
#         self.LM.to(self.device)
#         train_dataset = train_dataset.map(get_embedding, batched=True, batch_size=512)
#         valid_dataset = valid_dataset.map(get_embedding, batched=True, batch_size=512)
#         test_dataset = test_dataset.map(get_embedding, batched=True, batch_size=512)
#         # unload model from GPU 
#         self.LM.to("cpu")
        
#         train_dataset.save_to_disk(f"{self.data_path}/train_{self.data_info}.hf")
#         valid_dataset.save_to_disk(f"{self.data_path}/valid_{self.data_info}.hf")
#         test_dataset.save_to_disk(f"{self.data_path}/test_{self.data_info}.hf")
#         return 0


#     def train(self):
#         writer = SummaryWriter(f'{self.data_path}/runs/')
#         neural_predicates  = NeuralKG(self.embedding_length)
#         neural_predicates.to(self.device)
        
#         train_dataset = load_from_disk(f"{self.data_path}/train_{self.data_info}.hf")
#         valid_dataset = load_from_disk(f"{self.data_path}/valid_{self.data_info}.hf")
#         test_dataset = load_from_disk(f"{self.data_path}/test_{self.data_info}.hf")
        

#         train_dataset.set_format('torch', columns=["triple","label"], device = self.device)
#         valid_dataset.set_format('torch', columns=["triple","label"], device = self.device)
#         test_dataset.set_format('torch', columns=["triple","label"], device = self.device)
        
#         # def collate_fn(examples):
#         #     tuple_list = [i["triple"] for i in examples]
#         #     labels = [i["label"] for i in examples]
#         #     tuple_list = torch.tensor(tuple_list)
#         #     labels = torch.tensor(labels)
#         #     return {"tuple": (tuple_list), "label": (labels)}
#         # train_x = train_dataset["triple"]
#         # train_y = train_dataset["label"]
#         # valid_x = valid_dataset["triple"]
#         # valid_y = valid_dataset["label"]
#         # test_x = test_dataset["triple"]
#         # test_y = test_dataset["label"]

#         # new_data_train = torch.utils.data.TensorDataset(train_x, train_y)
#         # new_data_valid = torch.utils.data.TensorDataset(valid_x, valid_y)
#         # new_data_test = torch.utils.data.TensorDataset(test_x, test_y)
        
#         # train_dataloder = torch.utils.data.DataLoader(new_data_train, batch_size=self.bath_size, shuffle=True)
#         # valid_dataloader = torch.utils.data.DataLoader(new_data_valid, batch_size=self.bath_size, shuffle=True)
#         # test_dataloader = torch.utils.data.DataLoader(new_data_test, batch_size=self.bath_size, shuffle=True)
        
#         #load data
#         train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=self.bath_size, shuffle=True, )
#         valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.bath_size, shuffle=True, )
#         test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.bath_size, shuffle=True, )
        
#         # define the optimizer
#         optimizer = torch.optim.Adam(neural_predicates.parameters(), lr=0.001)
#         loss_function = nn.BCELoss()
        
#         # make tensorboard writer
#         print("Start Training")
#         for _ in tqdm.tqdm(range(self.epoch)):
#             neural_predicates.train() 
#             running_loss = 0.0
#             correct_train = 0
#             total = 0
#             all_tn, all_fp, all_fn, all_tp = 0, 0, 0, 0
#             for data in (train_dataloder):
#                 inputs = data["triple"]
#                 labels = data["label"]
#                 # set labels to float
#                 optimizer.zero_grad()
#                 outputs = neural_predicates(inputs)
#                 outputs = outputs.squeeze()
#                 loss = loss_function(outputs, labels.float())
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
#                 # check precision 
#                 correct_train += (outputs > 0.5).eq(labels > 0.5).sum().item()
#                 total += (labels.size(0))
#                 tn, fp, fn, tp = confusion_matrix(labels.cpu().detach().numpy().flatten(), outputs.cpu().detach().numpy().flatten() > 0.5).ravel()
#                 all_tn += tn
#                 all_fp += fp
#                 all_fn += fn
#                 all_tp += tp
#             average_loss = running_loss / len(train_dataloder)
#             acc = correct_train/total
#             precision = all_tp / (all_tp + all_fp + 1e-12)
#             recall = all_tp / (all_tp + all_fn + 1e-12)
#             f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
#             total_positive = all_tp + all_fn
#             print(f"[Train] Epoch: {_}, Loss: {average_loss}, Acc: {acc} Total Pos: {total_positive}, Precision: {precision}, Recall: {recall}, F1: {f1}")
#             # validate the model
#             writer.add_scalar("Loss/train", average_loss, _)
#             writer.add_scalar("Acc/train", acc, _)
#             writer.add_scalar("Precision/train", precision, _)
#             writer.add_scalar("Recall/train", recall, _)
#             writer.add_scalar("F1/train", f1, _)
#             writer.add_scalar("Total_Pos/train", total_positive, _)
            
#             valid_loss = 0.0
#             correct_valid = 0
#             total = 0
#             neural_predicates.eval()     # Optional when not using Model Specific layer
#             all_tn, all_fp, all_fn, all_tp = 0, 0, 0, 0
#             for i, data in enumerate(valid_dataloader):
#                 inputs = data["triple"]
#                 labels = data["label"]
#                 with torch.no_grad():
#                     outputs = neural_predicates(inputs)
#                     outputs = outputs.squeeze()
#                 loss = loss_function(outputs,labels.float())
#                 valid_loss += loss.item() 
#                 correct_valid += (outputs > 0.5).eq(labels > 0.5).sum().item()
#                 total += (labels.size(0))
#                 tn, fp, fn, tp = confusion_matrix(labels.cpu().detach().numpy().flatten(), outputs.cpu().detach().numpy().flatten() > 0.5).ravel()
#                 all_tn += tn
#                 all_fp += fp
#                 all_fn += fn
#                 all_tp += tp
#             precision = all_tp / (all_tp + all_fp + 1e-12)
#             recall = all_tp / (all_tp + all_fn + 1e-12)
#             f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
#             total_positive = all_tp + all_fn
#             acc_valid = correct_valid/total
#             average_valid_loss = valid_loss / len(valid_dataloader)
#             print(f"[Valid] Epoch: {_}, Loss: {average_valid_loss}, Acc: {acc_valid}, Total Pos: {total_positive}, Precision: {precision}, Recall: {recall}, F1: {f1}")
#             writer.add_scalar("Loss/valid", average_valid_loss, _)
#             writer.add_scalar("Acc/valid", acc_valid, _)
#             writer.add_scalar("Precision/valid", precision, _)
#             writer.add_scalar("Recall/valid", recall, _)
#             writer.add_scalar("F1/valid", f1, _)
#             writer.add_scalar("Total_Pos/valid", total_positive, _)
            
#             # set early stop 
#             if self.early_stop.early_stop(average_valid_loss):
#                 print("[Early Stopping Training]")
#                 break
#         writer.close()
        
#         # test on test data 
#         test_loss = 0.0
#         correct = 0
#         total = 0
#         neural_predicates.eval()     # Optional when not using Model Specific layer
#         all_tn, all_fp, all_fn, all_tp = 0, 0, 0, 0
#         for i, data in enumerate(test_dataloader):
#             inputs = data["triple"]
#             labels = data["label"]
#             with torch.no_grad():
#                 outputs = neural_predicates(inputs)
#                 outputs = outputs.squeeze()
#             loss = loss_function(outputs,labels.float())
#             test_loss += loss.item()
#             # compute acc 
#             correct += (outputs > 0.5).eq(labels > 0.5).sum().item()
#             total += (labels.size(0))
#             tn, fp, fn, tp = confusion_matrix(labels.cpu().detach().numpy().flatten(), outputs.cpu().detach().numpy().flatten() > 0.5).ravel()
#             all_tn += tn
#             all_fp += fp
#             all_fn += fn
#             all_tp += tp
#         precision = all_tp / (all_tp + all_fp + 1e-12)
#         recall = all_tp / (all_tp + all_fn + 1e-12)
#         f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
#         total_positive = all_tp + all_fn
#         acc_test = correct/total
#         average_test_loss = test_loss / len(test_dataloader)
#         print(f"[TEST] Accuracy: {acc_test}")
#         print(f"[TEST] Average Test Loss: {average_test_loss}")
#         print(f"[TEST] Precision: {precision}")
#         print(f"[TEST] Recall: {recall}")
#         print(f"[TEST] F1: {f1}")
#         print(f"[TEST] Total Pos: {total_positive}")
        
        
#         # save the model             
#         torch.save(neural_predicates.state_dict(), f"{self.data_path}/neural_predicates_{self.data_info}_{acc_test}.pt")

#         return neural_predicates 
    
#     def inference(self, test_data = []):
#         # load model 
#         neural_predicates = NeuralKG(self.embedding_length)
#         neural_predicates.load_state_dict(torch.load(f"{self.data_path}/neural_predicates_{self.data_info}.pt"))
#         neural_predicates.eval()
#         neural_predicates.to(self.device)
#         test_data_embeddings = self.make_single_embeddings(test_data)
#         test_data_embeddings = test_data_embeddings.to(self.device)
#         outputs = neural_predicates(test_data_embeddings)
#         threshold_outputs = (outputs > 0.5).cpu().detach().numpy().flatten()
#         # insert emoji 
#         symbol_outputs = [u'\u2713' if i == 1 else u'\u10102' for i in threshold_outputs]
#         for i, data in enumerate(test_data):
#             print(f"[Prediction] x: {data}, y: {symbol_outputs[i]}, prob: {outputs[i].item()}")
#         return outputs

        
        
    

        
if __name__ == "__main__":
    current_dir = os.getcwd()
    folder_name = '/gammaILP/'
    current_dir = current_dir + folder_name
    task_name = 'icews14'
    data_info = 'balance'
    train_path = f"{current_dir}/cache/{task_name}/triple_train_{data_info}.csv"
    valid_path = f"{current_dir}/cache/{task_name}/triple_train_{data_info}.csv"
    test_path = f"{current_dir}/cache/{task_name}/triple_train_{data_info}.csv"
    
    # model = DifferentiablePropositionalization(train_path, valid_path, test_path, epoch=20, batch_size =30480, data_info=data_info)
    # model.train()
    # model.inference(['Benjamin_Netanyahu$Make_a_visit$Xi','China$Engage_in_diplomatic_cooperation$South_Korea'])
    
    model = Distance_predicate(train_path, valid_path, test_path, epoch=3, batch_size =32, data_info=data_info, fine_tune = True)
    model.embedding_find_fine_tune()
    # model.inference_distance()
    # model.finetune()
    # model.inference(text='Benjamin_Netanyahu Make_a_visit [MASK]')