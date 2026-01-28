import pandas 
import os 
import sys 
# import the current path
current_dir = os.path.dirname(os.path.abspath(__file__))
task_name = 'icews14'
data_str = ['train', 'valid', 'test']
for data_type in data_str:
    data = f"{current_dir}/cache/{task_name}/triple_{data_type}_occ.csv"
    df = pandas.read_csv(data)
    print(f"{data_type} data: {df.shape}")
    # postive numbres 
    pos = df[df['label'] == 1]
    print(f"postive data: {pos.shape}")
    # negative numbers
    neg = df[df['label'] == 0]
    # cut the positive and negative data be same 
    neg = neg.sample(n=pos.shape[0])
    print(f"negative data: {neg.shape}")
    # combine the positive and negative data
    df = pandas.concat([pos, neg])
    print(f"combine data: {df.shape}")
    # shuffle the data
    df = df.sample(frac=1)
    # save the data
    df.to_csv(f"{current_dir}/cache/{task_name}/triple_{data_type}_balance.csv", index=False)
    print(f"save the balance data: {data_type}")
