# import torch 
# import numpy as np
# a = torch.tensor([[1e-10, 2e10],[5,6]])
# b = torch.tensor([[5, 6],[7,8],[1e-10,2e10],[3,4]])

# a = a.unsqueeze(1)
# print(a)
# b = b.unsqueeze(0)
# print(b)
# c = (a==b).all(dim=2).any(dim=1)
# print(c)


# z = torch.tensor([[0,1,0],[1,0,1]])
# indices = torch.nonzero(z == 1  , as_tuple=False)
# print("Indices:", indices)





# import torch

# x = torch.tensor([
#     [3, 4],
#     [1, 2],
#     [7,8],
#     [1, 2],
#     [5, 6],
#     [3, 4]
# ])

# # Step 1: Track unique rows and map them to indices
# seen = {}
# unique_rows = []
# inverse_indices = []

# for row in x:
#     row_tuple = tuple(row.tolist())
#     if row_tuple not in seen:
#         seen[row_tuple] = len(seen)  # assign new index
#         unique_rows.append(row)
#     inverse_indices.append(seen[row_tuple])

# # Step 2: Convert to tensors
# unique_tensor = torch.stack(unique_rows)               # (num_unique, D)
# inverse_tensor = torch.tensor(inverse_indices)         # (N,)

# print("Original tensor:\n", x)
# print("Unique rows (first occurrence order):\n", unique_tensor)
# print("Inverse indices:\n", inverse_tensor)


# import torch
# import torch.nn.functional as F

# x = torch.tensor([0.011, 0.012, 0.002, 0.001, 0.0002])

# # Add epsilon to avoid log(0)
# eps = 1e-6
# x = x.clamp(eps, 1 - eps)
# logit = torch.log(x / (1 - x))  # inverse sigmoid

# # Normalize back to 0-1 range
# scaled = (logit - logit.min()) / (logit.max() - logit.min())

# print(scaled)

# c = 'C'
# print(c.lower())



# from ucimlrepo import fetch_ucirepo 
  
# # fetch dataset 
# statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# # data (as pandas dataframes) 
# X = statlog_german_credit_data.data.features 
# y = statlog_german_credit_data.data.targets 

# # metadata 
# print(statlog_german_credit_data.metadata) 

# # variable information 
# print(statlog_german_credit_data.variables) 

#%%
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from scipy.optimize import linear_sum_assignment
# import numpy as np

# def plot_cluster_label_confusion(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
    
#     # Use Hungarian algorithm to reorder clusters
#     row_ind, col_ind = linear_sum_assignment(-cm)
#     cm = cm[:, col_ind]

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=[f'Cluster {i}' for i in col_ind],
#                 yticklabels=[f'Label {i}' for i in np.unique(y_true)])
#     plt.xlabel("Predicted Clusters")
#     plt.ylabel("True Labels")
#     plt.title("Confusion Matrix between Clusters and True Labels")
#     plt.show()
# plot_cluster_label_confusion([0, 1, 2, 0, 1, 2], [0, 0, 2, 1, 0, 2])
# %%

# print the mnist data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

mnist_dataset = datasets.MNIST(root='gammaILP/image/', train=True, download=False)

def convert(img,label, i ):
    # convert the whilte pixel in the image into transparent 
    # Get all pixel data
    #save the orginal image
    pixels = img.getdata()

    # Create a new list for modified pixels
    new_pixels = []
    inverse_new_pixels = []

    # Define a threshold to detect "white"
    threshold = 240

    for pixel in pixels:
        r, g, b, a = pixel
        if not(r > threshold and g > threshold and b > threshold):
            # If pixel is close to white, make it transparent
            new_pixels.append((255, 255, 255, 0))
            inverse_new_pixels.append((255,255,255,0))
        else:
            # Keep original pixel
            new_pixels.append((r, g, b, a))
            inverse_new_pixels.append((255-r, 255-g, 255-b, a))

    # Apply new data
    img.putdata(new_pixels)
    img.save(f'gammaILP/image/png/{label}_{i}.png')
    img.putdata(inverse_new_pixels)
    img.save(f'gammaILP/image/png/inverse_{label}_{i}.png')
    return img 


# print some data set with labels 
for i in range(150):
    image, label = mnist_dataset[i]
    plt.imshow(image, cmap='gray_r',)
    plt.axis('off')
    plt.show()
    
    # save the iamge 
    image.save(f'gammaILP/image/png/original_{label}_{i}.png')
    
    ##transfer into transparent format 
    # image = image.convert("RGBA")
    # image = convert(image,label,i)
    # plt.clf()





# %%
