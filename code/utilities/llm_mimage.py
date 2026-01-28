# %%
import sys
import os 
import pandas as pd
import random
import string


target_predicate = 'fizz'
task_name = 'fizz_m'
relation_arity = {'zero':1, 'succ':2, 'odd':1, 'even':1, 'pre':2,'lessthan':2,'buzz':1, 'fizz':1}

# %%
# read the training file 
train_data = pd.read_csv(f'gammaILP/cache/{task_name}/train_embeddings.csv', header=None)
# read all data in to list 
train_data_list = train_data.values.tolist()[1:]
# get all entities. 
entities = []
annotations = []
fact_with_annotations = []
relations_to_random = {}
for row in train_data_list:
    entities.append(int(row[1]))
    entities.append(int(row[2]))
    # Generate a random annotation of length 3
    annotation_1 = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    annotation_2 = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    annotations.append(annotation_1)
    annotations.append(annotation_2)
    relations = row[0]
    if relations not in relations_to_random:
        annotation_res = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
        relations_to_random[relations] = annotation_res
    if relation_arity[relations] == 1:
        fact_with_annotations.append(relations_to_random[relations]+'('+annotation_1+')')
        fact_with_annotations.append(relations_to_random[relations]+'('+annotation_2+')')
    elif relation_arity[relations] == 2:
        fact_with_annotations.append(relations_to_random[relations]+'('+annotation_1+','+annotation_2+')')
entities
annotations
fact_with_annotations

# %%
# import mnist images 
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_mnist_images(target_label = [2,3]):
    # Load MNIST dataset
    transform = transforms.ToTensor()
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Filter indices for the selected label
    all_images = []
    # Iterate through the dataset and collect images with the target label
    # If target_label is empty, return all images
    
    for index, image_label in enumerate(mnist_dataset):
        image = image_label[0]
        label = image_label[1]  # Get the label of the image
        if len(target_label) == 0:
            break
        if label == target_label[0]:
            all_images.append(image)
            target_label = target_label[1:]
    return all_images


# %%
# get 
all_iamges = load_mnist_images(target_label=entities)

# %%
all_fact_str = ','.join(fact_with_annotations)
all_fact_str

# %%
prompt = f"If you have the following images and their annotations, you also have the fact set in r(a_1,a_2), which r indicates the relation between image annotated with a_1 and the image annotated with a_2. All facts are: {all_fact_str}. Can you learn first-order logic program to describe the '{relations_to_random[target_predicate]}' with only existing relations and images?"
if not os.path.exists(f'./mnist_r/{task_name}'):
    os.makedirs(f'./mnist_r/{task_name}')
for i in range(len(all_iamges)):
    plt.imshow(all_iamges[i].squeeze(), cmap='gray')
    plt.title(f"Annotation: {annotations[i]}")
    plt.axis('off')
    print(f"Annotation: {annotations[i]}")
    # plt.show()
    # save image to file
    plt.savefig(f'./mnist_r/{task_name}/{annotations[i]}.png')
    plt.cla()

with open(f'./mnist_r/{task_name}/0Aprompt.txt', 'w') as f:
    f.write(prompt)
    f.write('\n')
    f.write(str(relations_to_random))
    f.close()
print(prompt)
print(relations_to_random)