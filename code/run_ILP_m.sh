# python -u  gammaILP/code/main.py --target_predicate 'ue' --task 'ue' --number_variable 2 --target_variable_arrange 'X@Y' --target_variable_arrange_index '0,1' 

# for i in 'cons' 'father' 'gp' 'length' 'member' 'odd' 'related' 
# do
#     python -u  gammaILP/code/main.py --target_predicate ${i} --task ${i} --number_variable 3 --target_variable_arrange 'X@Y' --target_variable_arrange_index '0,1'
# done

# for i in 'even' 'gcor' 'iscycle' 'odd' 'tc' 
# do 
#     python -u  gammaILP/code/main.py --target_predicate ${i} --task ${i} --number_variable 3 --target_variable_arrange 'X@X' --target_variable_arrange_index '0,0'
# done


# for adjr task
# python -u  gammaILP/code/main.py --target_predicate 'adjr' --task 'adjr' --number_variable 2 --target_variable_arrange 'X@X' --target_variable_arrange_index '0,0'

# for uncle task and father task 
# python -u gammaILP/code/embeddings.py
# python -u  gammaILP/code/main.py --target_predicate 'uncle' --task 'uncle' --number_variable 3 --target_variable_arrange 'X@Y' 

# for length task 
# python -u  gammaILP/code/main.py --target_predicate 'length' --task 'length' --number_variable 4 --target_variable_arrange 'X@Y' --minimal_precision 0.1 --random_negative 100

# for fizz and buzz task
python -u  gammaILP/code/main.py --target_predicate 'fizz' --task 'fizz_m' --number_variable 5 --target_variable_arrange 'X@X' --minimal_precision 0.2 --random_negative 100 --epoch 100 --lr_rule 0.01 --data_format 'image' --substitution_method chain_random --output_file_name 'fizz_m_5_X@X_lr0.01_chain_random'

python -u  gammaILP/code/main.py --target_predicate 'buzz' --task 'buzz_m' --number_variable 6 --target_variable_arrange 'X@X' --minimal_precision 0.2 --random_negative 100 --epoch 100 --lr_rule 0.01 --data_format 'image' --substitution_method chain_random --output_file_name 'buzz_m_6_X@X_lr0.01_chain_random'

python -u  gammaILP/code/main.py --target_predicate 'fizz' --task 'fizz_m' --number_variable 5 --target_variable_arrange 'X@X' --minimal_precision 0.2 --random_negative 100 --epoch 100 --lr_rule 0.1 --data_format 'image' --substitution_method chain_random --output_file_name 'fizz_m_5_X@X_lr0.1_chain_random'

python -u  gammaILP/code/main.py --target_predicate 'buzz' --task 'buzz_m' --number_variable 6 --target_variable_arrange 'X@X' --minimal_precision 0.2 --random_negative 100 --epoch 100 --lr_rule 0.1 --data_format 'image' --substitution_method chain_random --output_file_name 'buzz_m_6_X@X_lr0.1_chain_random'

python -u  gammaILP/code/main.py --target_predicate 'fizz' --task 'fizz_m' --number_variable 5 --target_variable_arrange 'X@X' --minimal_precision 0.2 --random_negative 100 --epoch 100 --lr_rule 0.5 --data_format 'image' --substitution_method chain_random --output_file_name 'fizz_m_5_X@X_lr0.5_chain_random'

python -u  gammaILP/code/main.py --target_predicate 'buzz' --task 'buzz_m' --number_variable 6 --target_variable_arrange 'X@X' --minimal_precision 0.2 --random_negative 100 --epoch 100 --lr_rule 0.5 --data_format 'image' --substitution_method chain_random --output_file_name 'buzz_m_6_X@X_lr0.5_chain_random'

# python -u  gammaILP/code/main.py --target_predicate 'even' --task 'even_m' --number_variable 3 --target_variable_arrange 'X@X' --minimal_precision 0.2 --random_negative 100 --epoch 100 --lr_rule 0.5 --data_format 'image' --substitution_method chain_random

# python -u  gammaILP/code/main.py --target_predicate 'odd' --task 'odd_m' --number_variable 3 --target_variable_arrange 'X@X' --minimal_precision 0.2 --random_negative 100 --epoch 100 --lr_rule 0.5 --data_format 'image' --substitution_method chain_random

# python -u  gammaILP/code/main.py --target_predicate 'pre' --task 'pre_m' --number_variable 3 --target_variable_arrange 'X@Y' --minimal_precision 0.5 --random_negative 100 --epoch 100 --lr_rule 0.5 --data_format 'image' --substitution_method chain_random


# python -u  gammaILP/code/main.py --target_predicate 'lessthan' --task 'lessthan_m' --number_variable 3 --target_variable_arrange 'X@Y' --minimal_precision 0.5 --random_negative 100 --epoch 100 --lr_rule 0.5 --data_format 'image' --substitution_method chain_random









