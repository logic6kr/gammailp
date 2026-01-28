# README for $\gamma$ ILP

### Before Running

The environment can be found in code/requirements.txt.

The images under dat-kandinsky-patterns are same with image under the same name in the gammaILP-master folder. 

### Code $\gamma$ ILP
All code are under code directory. The main code is in main.py, which is used to learn logic programs from the data. 

1. Run gammaILP/code/embedding.py for preprocess the data

    1.1 For learning from relational MNIST 
    ```python
        generating_embeddings('mnist')
    ```
    1.2 For learning from relational images
    ```python
        generating_embeddings('relation_image')
    ```
    1.3 For learning from ILP data
    ```python
        generating_embeddings('ILP', task_name='length')
    ```
    1.4 For learning from sequence data
    ```python
        train_sequence = [0,5,1,5,2,5,3,5,4,5,5,5,6,5,7,5,8,5,9,5]
        # train_sequence2 = [0,1,2,3,4,5,6,7,8,9]
        target = [1,3,5,7,9,11,13,15,17,19]
        generating_embeddings('mnist_sequence', sequence=train_sequence, target = target)
    ``` 
    1.5 For learning from Kandisky patterns
    ```python
        # for kan pattern 
        generating_embeddings('kandinsky', task_name='onetriangle_4to1')
    ```

2. Run gammaILP/code/main.py to learn logic programs.
Essential arguments are presented as follows:
    - `--task_name`: Name of the task, e.g., 'uncle', 'mnist', 'kandinsky_onered_4to1'.
    - `--target_predicate`: The target predicate to learn, e.g., 'uncle', 'lessthan', 'target'.
    - `--data_format`: Format of the data, e.g., 'kg' for knowledge graph, 'image' for image data, 'pi' for predicate invention.
    - `--lr`: Learning rate for ILP models.
    - `--lr_dkm`: Learning rate for DKM when image data is considered.  
    - `--n_clusters`: Number of clusters for the model, default is 10.
    - `--n_epochs`: Number of epochs for training, default is 30.
    - `--target_variable_arrange`: The variable term order in the target predicate. X@Y for binary target predicate and X@X for uniary target predicate.
    - `--number_variable`: Total number of variables in the logic program, including the variables in the head atom. 


There are some examples below:
```shell
    python main.py --task_name 'uncle' --target_predicate 'uncle' --lr 0.01 --data_format 'kg'
```
<!-- ```shell
    python main.py --task_name 'mnist' --target_predicate 'lessthan' --lr 0.01 --data_format 'image'
``` -->

Run with variational ViT embeddings
```shell
    python main.py --task_name 'lessthan_m' --target_predicate 'lessthan' --lr 0.01 --data_format 'image'
```

Run with variational autoencoder embeddings
```shell
    python main.py --task_name 'lessthan_ae' --target_predicate 'lessthan' --lr 0.01 --data_format 'image'
```

```shell
    python main.py --task_name 'kandinsky_onered_4to1' --target_predicate 'target' --lr 0.01 --lr_dkm 0.01 --data_format 'pi'git push -u origin main

```
```shell
python -u  gammaILP/code/main.py --target_predicate 'length' --task 'length' --number_variable 4 --target_variable_arrange 'X@Y' --minimal_precision 0.1 --random_negative 100
```git push -u origin main



See some best argument settings in Reference section below. 

3. Run code/Evaluate.py to obtain MRR and HITS when link predictions. Specific the `--task_name` and `--target_predicate` to the task and target predicate you want to evaluate. 
```shell
    python gammaILP/code/Evaluate.py --task_name 'mnist' --target_predicate 'lessthan'
```

4. Before checking the precision and recall of rules, we need to build all fact file named all.pl including the training facts and test facts annotated with 'TEST' at the end. Can run with the following command:
```shell
    python gammaILP/code/utilities/data_transfer_build_test_ILP.py --task_name 'uncle'
```

5. Run compute_metric_rextrule.py to test the precision and recall on the tested data. 
```shell
    python gammaILP/code/compute_metric_textrule.py --target_predicate 'uncle' --task_name 'uncle'
```


### Reference 
The learning rate for the ILP data. The larger learning rate, the logic program would be more discrete.  
1. GP: lr 0.5, epoch 200
2. Length: lr 0.5, epoch 200, random_negative 100 number_variable 4
3. Fizz: lr = 0.05, epoch 200, random_negative 100, number_variable 4, target_variable_arrange 'X@X'
4. Buzz: lr = 0.05, epoch 200, random_negative 100, number_variable 4, target_variable_arrange 'X@X'
5. Rest Tasks: lr = 0.05

# Special Case:
Kandisky 4to1 indicate the ratio of the positive examples and the negatives are 4:1. 
The Kandisky50 indicate we choose the top 100 exampels with 50 positive and 50 negative examples in total. The ratio of the positive to negative instances is still 4:1.