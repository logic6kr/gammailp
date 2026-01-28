import os 
import json
import re
import pandas  as pd
import numpy as np


# Load the embeddings from the CSV file
data = pd.read_csv('gammaILP/cache/relational_images/train_embeddings.csv')
all_predicate = []
for index, row in data.iterrows():
    # Convert the string representation of the list to an actual list
    if row['textR'] not in all_predicate:
        all_predicate.append(row['textR'])

formatted_text_linux =  ''
for item  in all_predicate:
    formatted_text_linux += f"'{item}' "
with open('gammaILP/cache/relational_images/predicates.json', 'w') as f:
    # Write the list to the file in JSON format
    json.dump(formatted_text_linux, f)

