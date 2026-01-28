import torch
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor, ViTModel, BertTokenizer, BertModel
from torch.utils.data import DataLoader,Subset
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
import pickle
from kandy_precossing import main as kandy_main

class generating_embeddings:
    def __init__(self, data_type='mnist',task_name='lessthan', sequence= [], target=[],output_cache_name='', remove_relations = []):
        print(f"Generating embeddings for {data_type} with task name {task_name}...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Load BERT model znd tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # 1. Load Huggingface ViT model and feature extractor
        self.vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.output_cache_name = output_cache_name
        self.remove_relations = remove_relations
        # build images 
        if data_type == 'mnist':
            self.number_consider = 20 # number of training data
            self.number_test = 10
            self.file_path = 'gammaILP/image/'
            self.ouput_path  = 'gammaILP/cache/mnist/'
            self.embeddings, self.labels = self.generate_embeddings_mnist(self.number_consider + self.number_test)
            self.make_relational_data(range(0,self.number_consider),'train')
            self.make_relational_data((range(self.number_consider, self.number_consider + self.number_test)),'test')
            
        elif  'mnist_sequence' in data_type:
            self.file_path = 'gammaILP/image/'
            self.ouput_path  = f'gammaILP/cache/{data_type}/'
            if not os.path.exists(self.ouput_path):
                os.makedirs(self.ouput_path)
            self.embeddings, self.labels = self.get_sequence_embedding(sequence= sequence)
            self.make_sequence_data(self.embeddings, self.labels, target = target, data_type='train')
            self.make_sequence_data(self.embeddings, self.labels, data_type='test', target = target)
            
        elif data_type == 'ILP':
            self.file_path = f'gammaILP/ILPdata/{task_name}/train.pl'
            self.ouput_path  = f'gammaILP/cache/{task_name}/'
            if not os.path.exists(self.ouput_path):
                os.makedirs(self.ouput_path)
            self.make_ILP_data()
        
        elif data_type == 'relation_mnist':
            self.number_consider = 20 # number of training data
            self.number_test = 10
            self.file_path = 'gammaILP/image/'
            if len(remove_relations) > 0:
                self.ouput_path  = f'gammaILP/cache/{task_name}_ppi/'
            else:
                self.ouput_path  = f'gammaILP/cache/{task_name}_m/'
            KB_base, image_base = self.read_facts(task_name, remove_relations= remove_relations)
            self.make_mnist_relational_data(KB_base, image_base)
            
        elif data_type == 'relation_image':
            self.number_consider = -1
            train_range = range(0, 500)
            test_range = range(100, 200)
            self.file_path = 'mmkb/FB15K/FB15K_EntityTriples.txt'
            self.ouput_path  = f'gammaILP/cache/relational_images'
            if not os.path.exists(self.ouput_path):
                os.makedirs(self.ouput_path)
            self.relation_image(train_range, data_type='train')
            self.relation_image(test_range, data_type='test')
        
        elif data_type == 'kandinsky':
            # kandy_main(task_name=task_name, training_numbers= 800, testing_numbers=200)
            kandy_main(task_name=task_name, training_numbers= 40, testing_numbers=10)
            # kandy_main(task_name=task_name, training_numbers= 15, testing_numbers=15)
            train_true_path = f'gammaILP/dat-kandinsky-patterns/{task_name}/train/true/cropped_objects/'
            train_false_path = f'gammaILP/dat-kandinsky-patterns/{task_name}/train/false/cropped_objects/'
            self.ouput_path  = f'gammaILP/cache/kandinsky_{task_name}/'
            test_true_path = f'gammaILP/dat-kandinsky-patterns/{task_name}/test/true/cropped_objects/'
            test_false_path = f'gammaILP/dat-kandinsky-patterns/{task_name}/test/false/cropped_objects/'
            if not os.path.exists(self.ouput_path):
                os.makedirs(self.ouput_path)
            self.kandy_pattern(positive_samples=train_true_path, negative_samples=train_false_path, sub_data = 'train')
            self.kandy_pattern(positive_samples=test_true_path, negative_samples=test_false_path, sub_data='test')

        elif data_type == 'half_predicate_half_pi':
            self.file_path = 'gammaILP/image/'
            self.ouput_path  = f'gammaILP/cache/{self.output_cache_name}/'
            if not os.path.exists(self.ouput_path):
                os.makedirs(self.ouput_path)
            self.embeddings, self.labels = self.get_sequence_embedding(sequence= sequence)
            self.temporal_with_predicate_placeholder(self.embeddings, self.labels, target = target, data_type='train')

        print(f"Embeddings generated and saved to {self.ouput_path}")

    def make_mnist_relational_data(self, KB_base, image_base):
        kb_labels = []
        for item in KB_base:
            first_entity = str(item[0])
            second_entity = str(item[2])
            kb_labels.append(first_entity)
            kb_labels.append(first_entity)
            kb_labels.append(second_entity)
            kb_labels.append(second_entity)
        image_labels = []
        for item in image_base:
            first_entity = str(item[0])
            second_entity = str(item[2])
            image_labels.append(first_entity)
            image_labels.append(first_entity)
            image_labels.append(second_entity)
            image_labels.append(second_entity)
        all_entity_embeddings, all_labels_m = self.generate_mnist_image_with_labels(kb_labels, image_labels)
        # build training data 
        self.relation_train = {'Entity1':[], 'Relation':[], 'Entity2':[], 'textR': [], 'text1': [], 'text2': []}
        self.relation_test = {'Entity1':[], 'Relation':[], 'Entity2':[], 'textR': [], 'text1': [], 'text2': []}
        fact_index = 0
        for item in KB_base:
            first_entity = str(item[0])
            second_entity = str(item[2])
            relation = str(item[1]).replace('/','').replace('.','')
            embedding_first = all_entity_embeddings[fact_index]
            embedding_second = all_entity_embeddings[fact_index + 1]
            embedding_first_test = all_entity_embeddings[fact_index + 2]
            embedding_second_test = all_entity_embeddings[fact_index + 3]
            embedding_relation = self.generate_bert_embeddings(relation)
            self.relation_train['Entity1'].append(embedding_first)
            self.relation_train['Relation'].append(embedding_relation)
            self.relation_train['Entity2'].append(embedding_second)
            self.relation_train['textR'].append(relation)
            self.relation_train['text1'].append(first_entity)
            self.relation_train['text2'].append(second_entity)
            self.relation_test['Entity1'].append(embedding_first_test)
            self.relation_test['Relation'].append(embedding_relation)
            self.relation_test['Entity2'].append(embedding_second_test)
            self.relation_test['textR'].append(relation)
            self.relation_test['text1'].append(first_entity)
            self.relation_test['text2'].append(second_entity)
            fact_index += 1
        # save the training data to a csv file
        train_df = pd.DataFrame(self.relation_train)
        if not os.path.exists(self.ouput_path):
            os.makedirs(self.ouput_path)
        train_df.to_csv(self.ouput_path + 'train_embeddings.csv', index=False, columns=[ 'textR', 'text1', 'text2'])
        print("Relations saved to train_embeddings.csv")
        # save the test data to a csv file
        test_df = pd.DataFrame(self.relation_test)
        test_df.to_csv(self.ouput_path + 'test_embeddings.csv', index=False, columns=[ 'textR', 'text1', 'text2'])
        print("Relations saved to test_embeddings.csv")
        # save the embeddings to a tensor file
        train_data_E = torch.tensor(np.array(self.relation_train['Entity1']))
        train_data_R = torch.tensor(np.array(self.relation_train['Relation']))
        train_data_L = torch.tensor(np.array(self.relation_train['Entity2']))
        # Save the tensors to a file
        torch.save(train_data_E, f"{self.ouput_path}/train_balance_bert_embeddings_E_10.pt")
        torch.save(train_data_R, f"{self.ouput_path}/train_balance_bert_embeddings_R_10.pt")
        torch.save(train_data_L, f"{self.ouput_path}/train_balance_bert_embeddings_L_10.pt")
        # save the test data to a tensor file
        test_data_E = torch.tensor(np.array(self.relation_test['Entity1']))
        test_data_R = torch.tensor(np.array(self.relation_test['Relation']))
        test_data_L = torch.tensor(np.array(self.relation_test['Entity2']))
        # Save the tensors to a file
        torch.save(test_data_E, f"{self.ouput_path}/test_balance_bert_embeddings_E_10.pt")
        torch.save(test_data_R, f"{self.ouput_path}/test_balance_bert_embeddings_R_10.pt")
        torch.save(test_data_L, f"{self.ouput_path}/test_balance_bert_embeddings_L_10.pt")


        if len(self.remove_relations) > 0:
            self.image_index_1 = []
            self.image_index_2 = []
            # append images information into embeddings 
            # positive_images = []
            # save mnist images embeddings into positive_images_data
            fact_index = 0 
            image_index_1 = 0
            image_index_2 = 1
            for item in image_base:
                first_entity = str(item[0])
                second_entity = str(item[2])
                # positive_images.append(all_entity_embeddings[fact_index].numpy())
                # positive_images.append(all_entity_embeddings[fact_index + 1].numpy())
                self.image_index_1.append(image_index_1)
                self.image_index_2.append(image_index_2)
                image_index_1 += 2
                image_index_2 += 2
                fact_index += 4
            # torch.save(positive_images, f"{self.ouput_path}/positive_images_data_train.pt")
            all_image_index = self.image_index_1 + self.image_index_2
            # save all image index to a pickle file,  image order -> index based on tuple of entities
            with open(f"{self.ouput_path}/index_to_image_train.pkl", 'wb') as f:
                pickle.dump(all_image_index, f)
        return all_entity_embeddings, all_labels_m


        

    def kandy_pattern(self, positive_samples, negative_samples, sub_data='train'):
        # open all the images in the positive folder 
        positive_images = []
        positive_images_data = []
        negative_images = []
        negative_images_data = []
        index_to_image = []
        # the image in a folder under the positive sampels 
        for root, dirs, files in os.walk(positive_samples):
            single_data = []
            if len(files) != 5:
                continue
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    if 'preview' not in file:  # Exclude preview images
                        positive_images.append(os.path.join(root, file))
                        single_data.append(self.get_embeddings(os.path.join(root, file)))
                        index_to_image.append(os.path.join(root, file))
            if len(single_data) > 0:
                positive_images_data.append(single_data)
            
        # the image in a folder under the negative sampels
        for root, dirs, files in os.walk(negative_samples):
            single_data = []
            if len(files) != 5:
                continue
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    if 'preview' not in file:  # Exclude preview images
                        negative_images.append(os.path.join(root, file))
                        single_data.append(self.get_embeddings(os.path.join(root, file)))
                        index_to_image.append(os.path.join(root, file))
            if len(single_data) > 0:
                negative_images_data.append(single_data)

        # save the tensor to a file
        positive_images_data = torch.tensor(np.array(positive_images_data))
        negative_images_data = torch.tensor(np.array(negative_images_data))
        torch.save(positive_images_data, f"{self.ouput_path}/positive_images_data_{sub_data}.pt")
        torch.save(negative_images_data, f"{self.ouput_path}/negative_images_data_{sub_data}.pt")
        # save index to image mapping to pickle file
        with open(f"{self.ouput_path}/index_to_image_{sub_data}.pkl", 'wb') as f:
            pickle.dump(index_to_image, f)
        
    def temporal_with_predicate_placeholder(self, embeddings, labels, data_type='train', target = [0,2,4,6,8]):
        # todo 1: create embeddings files for known facts
        # create a pandas dataframe to store the embeddings if their label meet any of the predicate 
        # target function receive the binary tuple to logical conditions

        # make image index 
        self.image_index_1 = []
        self.image_index_2 = []


        #1. Define the predicates
        # only consider the 5 
        self.predicate = {'succ':lambda tup: tup[0] == tup[1]-1, 'zero':lambda tup: tup[0] == 0} 
        self.predicate_embeddings = {}
        # temporal relation is added by human 
        
        #2. Create the pandas codebook
        self.relation = {'Entity1':[], 'Relation':[], 'Entity2':[], 'textR': [], 'text1': [], 'text2': []}
        
        # Genetate the bert emebddings for each predicate 
        # import ber model 
        self.bert_model.eval()
        
        # update predicate name to embeddings dic 
        for key in self.predicate.keys():
            self.predicate_embeddings[key] = self.generate_bert_embeddings(key)
        
        # 3. Iterate through the range and check the predicates
        checked_tuple = set([])
        all_label_list = labels.tolist()
        for i in range(0,len(all_label_list)):
            for j in range(0,len(all_label_list)):
                first_label = all_label_list[i]
                second_label = all_label_list[j]
                print(f"Checking relation between {first_label} and {second_label}")
                
                if i in target:
                    if f'{first_label}_{first_label}_target' in checked_tuple:
                        pass
                    else:
                        self.relation['Entity1'].append(self.embeddings[i].numpy())
                        self.relation['Relation'].append(self.generate_bert_embeddings('target'))
                        self.relation['Entity2'].append(self.embeddings[i].numpy())
                        self.relation['textR'].append('target')
                        self.relation['text1'].append(first_label)
                        self.relation['text2'].append(first_label)
                        self.image_index_1.append(i)
                        self.image_index_2.append(i)
                        checked_tuple.add(f'{first_label}_{first_label}_target')
                
                # todo add succ and zero predicate into the relation
                # for key, func in self.predicate.items():
                #     # check if the tuple is already checked
                #     if func((first_label,second_label)):
                #         if key != 'zero' and key != 'target':
                #             if f'{first_label}_{second_label}_{key}' in checked_tuple:
                #                 continue
                #             self.relation['Entity1'].append(self.embeddings[i].numpy())
                #             self.relation['Relation'].append(self.predicate_embeddings[key])
                #             self.relation['Entity2'].append(self.embeddings[j].numpy())
                #             self.relation['textR'].append(key)
                #             self.relation['text1'].append(first_label)
                #             self.relation['text2'].append(second_label)
                #             checked_tuple.add(f'{first_label}_{second_label}_{key}')
                #         else:
                #             # Only add the zero relation once for each i
                #             if f'{first_label}_{first_label}_{key}' in checked_tuple:
                #                 continue
                #             self.relation['Entity1'].append(self.embeddings[i].numpy())
                #             self.relation['Relation'].append(self.predicate_embeddings[key])
                #             self.relation['Entity2'].append(self.embeddings[i].numpy())
                #             self.relation['textR'].append(key)
                #             self.relation['text1'].append(first_label)
                #             self.relation['text2'].append(first_label)
                #             checked_tuple.add(f'{first_label}_{first_label}_{key}')


        # add temporal information to the embeddings 
        self.relation['Entity1'].append(embeddings[0])
        self.relation['Relation'].append(self.generate_bert_embeddings('start'))
        self.relation['Entity2'].append(embeddings[0])
        self.relation['textR'].append('start')
        self.relation['text1'].append(str(labels[0].item()))
        self.relation['text2'].append(str(labels[0].item()))
        self.image_index_1.append(0)
        self.image_index_2.append(0)
        
        # add temporal information to the embeddings
        for i in range(0, len(embeddings)):
            for j in range(i+1, len(embeddings)):
                length = j - i
                if f'{all_label_list[i]}_{all_label_list[j]}_before_{length}' in checked_tuple:
                    pass
                else:
                    self.relation['Entity1'].append(embeddings[i])
                    self.relation['Relation'].append(self.generate_bert_embeddings(f'before_{length}'))
                    self.relation['Entity2'].append(embeddings[j])
                    self.relation['textR'].append(f'before_{length}')
                    self.relation['text1'].append(str(all_label_list[i]))
                    self.relation['text2'].append(str(all_label_list[j]))
                    checked_tuple.add(f'{all_label_list[i]}_{all_label_list[j]}_before_{length}')
                    self.image_index_1.append(i)
                    self.image_index_2.append(j)
        
        # 4. Convert to DataFrame
        df = pd.DataFrame(self.relation)
        print(df.head())
        # 5. Save the DataFrame to a CSV file
        df.to_csv(self.ouput_path + f'{data_type}_embeddings.csv', index=False, columns=[ 'textR', 'text1', 'text2'])
        print("Relations saved to relations.csv")
        # 6. Save the embeddings to a tensor file 
        train_data_E = torch.tensor(self.relation['Entity1'])
        train_data_R = torch.tensor(self.relation['Relation'])
        train_data_L = torch.tensor(self.relation['Entity2'])
        # Save the tensors to a file
        torch.save(train_data_E, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_E_10.pt")
        torch.save(train_data_R, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_R_10.pt")
        torch.save(train_data_L, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_L_10.pt")


        # append images information into embeddings 
        positive_images = []
        # save mnist images embeddings into positive_images_data
        for i in range(0, len(all_label_list)):
            label = all_label_list[i]
            positive_images.append(self.embeddings[i].numpy())
        torch.save(positive_images, f"{self.ouput_path}/positive_images_data_train.pt")
        all_image_index = self.image_index_1 + self.image_index_2
        # save all image index to a pickle file
        #! record the indexes of image 1 and image 2 stored in train.csv tuple fact from mnist_train.pkl
        with open(f"{self.ouput_path}/index_to_image_train.pkl", 'wb') as f:
            pickle.dump(all_image_index, f)
        return 0 



    def relation_image(self, indec=None, data_type='train'):
        # 1. Load Huggingface ViT model and feature extractor
        self.vision_model.eval()
        
        # 2. open the relational data 
        all_data = []
        with open(self.file_path , 'r') as f:
            for line in f:
                # split the line into a list of strings
                line = line.replace('\n','')
                line = line.split(' ')
                all_data.append((line[0][1:].replace('/','.'), line[1], line[2][1:].replace('/','.')))
            f.close()
        
        # build the dataset 
        relation_df = {'textR': [], 'text1': [], 'text2': []}
        train_data_E = []
        train_data_R = []
        train_data_L = []
        if indec != None:
            consider_all_data = [all_data[i] for i in indec]
        else:
            consider_all_data = all_data.copy()
        for item in tqdm(consider_all_data):
            first_entity = str(item[0])
            second_entity = str(item[2])
            relation = str(item[1]).replace('/','').replace('.','')
            try:
                embedding_first = self.generate_image_embeddings(first_entity)
                embedding_second = self.generate_image_embeddings(second_entity)
            except:
                print(f"Error loading image for {first_entity} or {second_entity}. Skipping...")
                continue
            embedding_relation = self.generate_bert_embeddings(relation)
            relation_df['textR'].append(relation)
            relation_df['text1'].append(first_entity)
            relation_df['text2'].append(second_entity)
            train_data_R.append(embedding_relation)
            train_data_E.append(embedding_first)
            train_data_L.append(embedding_second)
            
        # 4. Convert to DataFrame
        df = pd.DataFrame(relation_df)
        print(df.head())
        # 5. Save the DataFrame to a CSV file
        df.to_csv(self.ouput_path + f'/{data_type}_embeddings.csv', index=False, columns=['textR', 'text1', 'text2'])
        print("Relations saved to relations.csv")
        # 6. Save the embeddings to a tensor file
        train_data_E = torch.tensor(np.array(train_data_E))
        train_data_R = torch.tensor(np.array(train_data_R))
        train_data_L = torch.tensor(np.array(train_data_L))
        # Save the tensors to a file
        torch.save(train_data_E, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_E_10.pt")
        torch.save(train_data_R, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_R_10.pt")
        torch.save(train_data_L, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_L_10.pt")
    
    def get_embeddings(self, image_path):
        # 1. Open and transform the image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet normalization
                                std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = self.vision_model(img_tensor)
            # Extract CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.squeeze(0)

    def generate_mnist_image_with_labels(self, label_list, partial_labels):

        self.vision_model.eval()

        # 2. Prepare the MNIST dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize to 224x224
            transforms.Grayscale(num_output_channels=3),  # convert 1 channel to 3 channels (RGB fake)
            transforms.ToTensor(),  # convert to tensor
        ])

        mnist_dataset = datasets.MNIST(root=self.file_path, train=True, download=False, transform=transform)
        # only consider top 10 instances 
        # indices = range(0, all_data_length)
        # mnist_dataset = Subset(mnist_dataset, indices)
        
        # Create a DataLoader for the MNIST dataset
        mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)

        # 3. Extract embeddings in batch
        all_embeddings = []
        all_labels = []
        all_images_train = []
        count_train = 0
        stored_embeddings = []
        with torch.no_grad():
            for images, labels in mnist_loader:
                if len(label_list) == 0:
                    break
                # transfer labels to string
                labels = str(labels.item())
                if labels != label_list[0]:
                    continue
                # ViT expects pixel values normalized between [0,1] already
                inputs = {'pixel_values': images}
                
                # Pass through the ViT model
                outputs = self.vision_model(**inputs)
                
                # Extract CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
                
                all_embeddings.append(embeddings)
                all_labels.append(labels)
                label_list = label_list[1:]  # Remove the first label after processing

            
            for images, labels in mnist_loader:
                if len(partial_labels) == 0:
                    break
                # transfer labels to string
                labels = str(labels.item())
                if labels != partial_labels[0]:
                    continue
                # ViT expects pixel values normalized between [0,1] already
                inputs = {'pixel_values': images}
                
                # Pass through the ViT model
                outputs = self.vision_model(**inputs)
                
                # Extract CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
                stored_embeddings.append(embeddings)
                partial_labels = partial_labels[1:]  # Remove the first label after processing
                if count_train < 2:
                    all_images_train.append(images)
                elif count_train == 3:
                    count_train = 0 
                    continue
                count_train += 1

        # 4. Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, 768)
        # all_labels = torch.cat(all_labels, dim=0)  # (N)

        print(f"Final embeddings shape: {all_embeddings.shape}")  # (60000, 768)
        # print(f"Final labels shape: {all_labels.shape}")          # (60000,)
        print('All labels are', all_labels)
        # save image to a pickle file
        stored_embeddings = torch.cat(stored_embeddings, dim=0)
        torch.save(stored_embeddings, f"{self.ouput_path}/positive_images_data_train.pt")
        with open(f"{self.ouput_path}/mnist_images.pkl", 'wb') as f:
            pickle.dump(all_images_train, f)
            f.close()
        return all_embeddings, all_labels

    
    def generate_image_embeddings(self, path):
        # generate the image embeddings based on the path 
        self.root_path = 'mmkb/image-graph_images/'
        self.image_path = self.root_path + path
        
        # choose random image under the self.image_path
        
        # 2. Get a random image from the folder
        image_files = [f for f in os.listdir(self.image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random_image_path = os.path.join(self.image_path, random.choice(image_files))
        
        # 3. Open and transform the image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet normalization
                                std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(random_image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  #
        with torch.no_grad():
            outputs = self.vision_model(img_tensor)
            # Extract CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.squeeze(0)


    
    def get_sequence_embedding(self, sequence = []):
        self.vision_model.eval()

        # 2. Prepare the MNIST dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize to 224x224
            transforms.Grayscale(num_output_channels=3),  # convert 1 channel to 3 channels (RGB fake)
            transforms.ToTensor(),  # convert to tensor
        ])

        mnist_dataset = datasets.MNIST(root=self.file_path, train=True, download=False, transform=transform)
        # only consider top 10 instances 
        indices = range(0, 1000)
        mnist_dataset = Subset(mnist_dataset, indices)
        
        # Create a DataLoader for the MNIST dataset
        mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=False)

        # 3. Extract embeddings in batch
        all_embeddings = []
        all_labels = []
        all_images = []
        i = 0
        with torch.no_grad():
            for images, labels in mnist_loader:
                if i >= len(sequence):
                    break
                if labels.item() != sequence[i]:
                    continue
                # ViT expects pixel values normalized between [0,1] already
                inputs = {'pixel_values': images}
                
                # Pass through the ViT model
                outputs = self.vision_model(**inputs)
                
                # Extract CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
                
                all_embeddings.append(embeddings)
                all_labels.append(labels)
                all_images.append(images)
                i += 1

        # 4. Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, 768)
        all_labels = torch.cat(all_labels, dim=0)  # (N)

        print(f"Final embeddings shape: {all_embeddings.shape}")  # (60000, 768)
        print(f"Final labels shape: {all_labels.shape}")          # (60000,)
        print('All labels are', all_labels)
        # save image to a pickle file 
        with open(f"{self.ouput_path}/mnist_images.pkl", 'wb') as f:
            pickle.dump(all_images, f)
            f.close()
        return all_embeddings, all_labels
    
    def generate_embeddings_mnist(self, all_data_length = 50):
        
        self.vision_model.eval()

        # 2. Prepare the MNIST dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # resize to 224x224
            transforms.Grayscale(num_output_channels=3),  # convert 1 channel to 3 channels (RGB fake)
            transforms.ToTensor(),  # convert to tensor
        ])
        
        mnist_dataset = datasets.MNIST(root=self.file_path, train=True, download=False, transform=transform)
        # only consider top 10 instances 
        indices = range(0, all_data_length)
        mnist_dataset = Subset(mnist_dataset, indices)
        
        # Create a DataLoader for the MNIST dataset
        mnist_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=False)

        # 3. Extract embeddings in batch
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in mnist_loader:
                # ViT expects pixel values normalized between [0,1] already
                inputs = {'pixel_values': images}
                
                # Pass through the ViT model
                outputs = self.vision_model(**inputs)
                
                # Extract CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
                
                all_embeddings.append(embeddings)
                all_labels.append(labels)

        # 4. Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, 768)
        all_labels = torch.cat(all_labels, dim=0)  # (N)

        print(f"Final embeddings shape: {all_embeddings.shape}")  # (60000, 768)
        print(f"Final labels shape: {all_labels.shape}")          # (60000,)
        print('All labels are', all_labels)
        return all_embeddings, all_labels

    def read_facts(self, task_name, remove_relations=[]):
        # load facts from the file
        KB_base= []
        image_base = []
        data_source = 'gammaILP/ILPdata/' + task_name + '/train.pl'
        with open(data_source, 'r') as f:
            for line in f:
                # split the line into a list of strings
                line = line.replace('\n','')
                line = line.split(' ')
                image_base.append(line)
                if line[1] in remove_relations:
                    continue
                KB_base.append(line)
            f.close()
        # image base include all facts 
        # KB_based inlcude only the facts that are not in remove_relations
        return KB_base, image_base

    def read_ILP(self):
        # load the lp data 
        data = []
        with open(self.file_path , 'r') as f:
            for line in f:
                # split the line into a list of strings
                line = line.replace('\n','')
                line = line.split(' ')
                data.append(line)
            f.close()
        return data
    
    # create a function to generate the embeddings
    def generate_bert_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    def make_ILP_data(self):
        data = self.read_ILP()
        # Move the model to the device
        self.bert_model.eval()
        #1. Define the predicates

        self.predicate_embeddings = {}
        
        #2. Create the pandas codebook
        self.relation = {'Entity1':[], 'Relation':[], 'Entity2':[], 'textR': [], 'text1': [], 'text2': []}
        
        # Genetate the bert emebddings for each predicate 
        # import ber model 

        # update predicate name to embeddings dic 
        for item in tqdm(data):
            first_entity = str(item[0])
            second_entity = str(item[2])
            relation = str(item[1])
            embedding_first = self.generate_bert_embeddings(first_entity)
            embedding_second = self.generate_bert_embeddings(second_entity)
            embedding_relation = self.generate_bert_embeddings(relation)
            self.relation['Entity1'].append(embedding_first)
            self.relation['Relation'].append(embedding_relation)
            self.relation['Entity2'].append(embedding_second)
            self.relation['textR'].append(relation)
            self.relation['text1'].append(first_entity)
            self.relation['text2'].append(second_entity)
            

        # 4. Convert to DataFrame
        df = pd.DataFrame(self.relation)
        print(df.head())
        # 5. Save the DataFrame to a CSV file
        df.to_csv(self.ouput_path + 'train_embeddings.csv', index=False, columns=[ 'textR', 'text1', 'text2'])
        print("Relations saved to relations.csv")
        # 6. Save the embeddings to a tensor file 
        train_data_E = torch.tensor(np.array(self.relation['Entity1']))
        train_data_R = torch.tensor(np.array(self.relation['Relation']))
        train_data_L = torch.tensor(np.array(self.relation['Entity2']))
        # Save the tensors to a file
        torch.save(train_data_E, f"{self.ouput_path}/train_balance_bert_embeddings_E_10.pt")
        torch.save(train_data_R, f"{self.ouput_path}/train_balance_bert_embeddings_R_10.pt")
        torch.save(train_data_L, f"{self.ouput_path}/train_balance_bert_embeddings_L_10.pt")

        return 


    def make_sequence_data(self, embeddings, labels, data_type='train', target = [0,2,4,6,8]):
        # create a pandas dataframe to store the embeddings if their label meet any of the predicate 
        # target function receive the binary tuple to logical conditions
        
        #1. Define the predicates
        # only consider the 5 
        self.predicate = {'succ':lambda tup: tup[0] == tup[1]-1, 'zero':lambda tup: tup[0] == 0} 
        self.predicate_embeddings = {}
        # temporal relation is added by human 
        
        
        #2. Create the pandas codebook
        self.relation = {'Entity1':[], 'Relation':[], 'Entity2':[], 'textR': [], 'text1': [], 'text2': []}
        
        # Genetate the bert emebddings for each predicate 
        # import ber model 
        self.bert_model.eval()
        
        # update predicate name to embeddings dic 
        for key in self.predicate.keys():
            self.predicate_embeddings[key] = self.generate_bert_embeddings(key)
        
        # 3. Iterate through the range and check the predicates
        checked_tuple = set([])
        all_label_list = labels.tolist()
        for i in range(0,len(all_label_list)):
            for j in range(0,len(all_label_list)):
                first_label = all_label_list[i]
                second_label = all_label_list[j]
                print(f"Checking relation between {first_label} and {second_label}")
                
                if i in target:
                    if f'{first_label}_{first_label}_target' in checked_tuple:
                        pass
                    else:
                        self.relation['Entity1'].append(self.embeddings[i].numpy())
                        self.relation['Relation'].append(self.generate_bert_embeddings('target'))
                        self.relation['Entity2'].append(self.embeddings[i].numpy())
                        self.relation['textR'].append('target')
                        self.relation['text1'].append(first_label)
                        self.relation['text2'].append(first_label)
                        checked_tuple.add(f'{first_label}_{first_label}_target')
                
                
                for key, func in self.predicate.items():
                    # check if the tuple is already checked
                    if func((first_label,second_label)):
                        if key != 'zero' and key != 'target':
                            if f'{first_label}_{second_label}_{key}' in checked_tuple:
                                continue
                            self.relation['Entity1'].append(self.embeddings[i].numpy())
                            self.relation['Relation'].append(self.predicate_embeddings[key])
                            self.relation['Entity2'].append(self.embeddings[j].numpy())
                            self.relation['textR'].append(key)
                            self.relation['text1'].append(first_label)
                            self.relation['text2'].append(second_label)
                            checked_tuple.add(f'{first_label}_{second_label}_{key}')
                        else:
                            # Only add the zero relation once for each i
                            if f'{first_label}_{first_label}_{key}' in checked_tuple:
                                continue
                            self.relation['Entity1'].append(self.embeddings[i].numpy())
                            self.relation['Relation'].append(self.predicate_embeddings[key])
                            self.relation['Entity2'].append(self.embeddings[i].numpy())
                            self.relation['textR'].append(key)
                            self.relation['text1'].append(first_label)
                            self.relation['text2'].append(first_label)
                            checked_tuple.add(f'{first_label}_{first_label}_{key}')
        # add temporal information to the embeddings 
        self.relation['Entity1'].append(embeddings[0])
        self.relation['Relation'].append(self.generate_bert_embeddings('start'))
        self.relation['Entity2'].append(embeddings[0])
        self.relation['textR'].append('start')
        self.relation['text1'].append(str(labels[0].item()))
        self.relation['text2'].append(str(labels[0].item()))
        
        # add temporal information to the embeddings
        for i in range(0, len(embeddings)):
            for j in range(i+1, len(embeddings)):
                length = j - i
                if f'{all_label_list[i]}_{all_label_list[j]}_before_{length}' in checked_tuple:
                    pass
                else:
                    self.relation['Entity1'].append(embeddings[i])
                    self.relation['Relation'].append(self.generate_bert_embeddings(f'before_{length}'))        
                    self.relation['Entity2'].append(embeddings[j])
                    self.relation['textR'].append(f'before_{length}')
                    self.relation['text1'].append(str(all_label_list[i]))
                    self.relation['text2'].append(str(all_label_list[j]))
                    checked_tuple.add(f'{all_label_list[i]}_{all_label_list[j]}_before_{length}')
        
        # 4. Convert to DataFrame
        df = pd.DataFrame(self.relation)
        print(df.head())
        # 5. Save the DataFrame to a CSV file
        df.to_csv(self.ouput_path + f'{data_type}_embeddings.csv', index=False, columns=[ 'textR', 'text1', 'text2'])
        print("Relations saved to relations.csv")
        # 6. Save the embeddings to a tensor file 
        train_data_E = torch.tensor(self.relation['Entity1'])
        train_data_R = torch.tensor(self.relation['Relation'])
        train_data_L = torch.tensor(self.relation['Entity2'])
        # Save the tensors to a file
        torch.save(train_data_E, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_E_10.pt")
        torch.save(train_data_R, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_R_10.pt")
        torch.save(train_data_L, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_L_10.pt")
    
    def make_relational_data(self, indices, data_type='train'):
        # create a pandas dataframe to store the embeddings if their label meet any of the predicate 
        
        #1. Define the predicates
        self.predicate = {'succ':lambda tup: tup[0] == tup[1]-1, 'lessthan':lambda tup: tup[0] < tup[1], 'zero':lambda tup: tup[0] == 0}
        self.predicate_embeddings = {}
        
        #2. Create the pandas codebook
        self.relation = {'Entity1':[], 'Relation':[], 'Entity2':[], 'textR': [], 'text1': [], 'text2': []}
        
        # Genetate the bert emebddings for each predicate 
        # import ber model 
        
        self.bert_model.eval()
        
        # update predicate name to embeddings dic 
        for key in self.predicate.keys():
            self.predicate_embeddings[key] = self.generate_bert_embeddings(key)

        
        # 3. Iterate through the range and check the predicates
        recorded_zero_index = []
        for i in indices:
            for j in indices:
                first_label = self.labels[i].item()
                second_label = self.labels[j].item()
                print(f"Checking relation between {first_label} and {second_label}")
                for key, func in self.predicate.items():
                    if func((first_label,second_label)):
                        if key != 'zero':
                            self.relation['Entity1'].append(self.embeddings[i].numpy())
                            self.relation['Relation'].append(self.predicate_embeddings[key])
                            self.relation['Entity2'].append(self.embeddings[j].numpy())
                            self.relation['textR'].append(key)
                            self.relation['text1'].append(first_label)
                            self.relation['text2'].append(second_label)
                        elif key == 'zero' and i not in recorded_zero_index:
                            # Only add the zero relation once for each i
                            self.relation['Entity1'].append(self.embeddings[i].numpy())
                            self.relation['Relation'].append(self.predicate_embeddings[key])
                            self.relation['Entity2'].append(self.embeddings[i].numpy())
                            self.relation['textR'].append(key)
                            self.relation['text1'].append(first_label)
                            self.relation['text2'].append(first_label)
                            recorded_zero_index.append(i)

        # 4. Convert to DataFrame
        df = pd.DataFrame(self.relation)
        print(df.head())
        # 5. Save the DataFrame to a CSV file
        df.to_csv(self.ouput_path + f'{data_type}_embeddings.csv', index=False, columns=[ 'textR', 'text1', 'text2'])
        print("Relations saved to relations.csv")
        # 6. Save the embeddings to a tensor file 
        train_data_E = torch.tensor(self.relation['Entity1'])
        train_data_R = torch.tensor(self.relation['Relation'])
        train_data_L = torch.tensor(self.relation['Entity2'])
        # Save the tensors to a file
        torch.save(train_data_E, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_E_10.pt")
        torch.save(train_data_R, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_R_10.pt")
        torch.save(train_data_L, f"{self.ouput_path}/{data_type}_balance_bert_embeddings_L_10.pt")

        
if __name__ == "__main__":
    # todo for relational minist
    # generating_embeddings('mnist')
    # task = sys.argv[1] if len(sys.argv) > 1 else 'uncle'
    # removed_relations = []
    # generating_embeddings('relation_mnist', task_name=task, remove_relations=removed_relations)

    # todo for ILP datasets
    # generating_embeddings('ILP', task_name=task)
    
    # todo for relation image
    # generating_embeddings('relation_image')
    
    
    # todo for image sequence 
    # train_sequence = [0,5,1,5,2,5,3,5,4,5,5,5,6,5,7,5,8,5,9,5]
    # # train_sequence2 = [0,1,2,3,4,5,6,7,8,9]
    # target = [1,3,5,7,9,11,13,15,17,19]
    # generating_embeddings('mnist_sequence', sequence=train_sequence, target = target)
    
    # todo for kan pattern 
    generating_embeddings('kandinsky', task_name='onetriangle_4to1')
    generating_embeddings('kandinsky', task_name='onered_4to1')
    generating_embeddings('kandinsky', task_name='twopairs_50')

    # todo for predicate as placeholder and known temporal relations
    # for temporal image sequence v2 without succ predicate 
    # train_sequence = [0,5,1,5,2,5,3,5,4,5,5,5,6,5,7,5,8,5,9,5]
    # # target = [1,3,5,7,9,11,13,15,17,19]
    # target = [0,2,4,6,8,12,14,16,18]
    # generating_embeddings(data_type='half_predicate_half_pi', sequence=train_sequence, target = target, output_cache_name = 'ppi_sequence_odd')
    
    
