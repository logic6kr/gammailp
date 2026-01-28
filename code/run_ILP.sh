# generate embeddings first

# for i in  'cons' 'father' 'gp' 'length' 'member' 'odd' 'related' 'ue' 'even' 'gcor' 'iscycle' 'tc' 'adjr' 'uncle' 'buzz' 'fizz' 'husband' 'lessthan' 'pre' 'son' 
# do
#     python -u gammaILP/code/embeddings.py ${i}
# done



# for i in 'cons' 'father' 'lessthan' 'member' 'related' 'ue' 'pre' 'son' 
# do
# for j in 1 2 3 4 5 6 7 8 9 10
# do
#     echo ${i}
#     python -u  gammaILP/code/main.py --target_predicate ${i} --task ${i} --number_variable 3 --target_variable_arrange 'X@Y'  --data_format 'kg'
# done
# done

# for i in 'even' 'gcor' 'iscycle' 'odd' 'tc' 
# do
# for j in 1 2 3 4 5 6 7 8 9 10
# do
#     echo ${i}
#     python -u  gammaILP/code/main.py --target_predicate ${i} --task ${i} --number_variable 3 --target_variable_arrange 'X@X'  --data_format 'kg'
# done
# done


# # for adjr task
# for j in 1 2 3 4 5 6 7 8 9 10
# do
# echo 'adjr'
# python -u  gammaILP/code/main.py --target_predicate 'adjr' --task 'adjr' --number_variable 2 --target_variable_arrange 'X@X'  --data_format 'kg'
# done 



# # # for gp task 
# for j in 1 2 3 4 5 6 7 8 9 10
# do
# echo 'gp'
# python -u  gammaILP/code/main.py --target_predicate 'gp' --task 'gp' --number_variable 3 --target_variable_arrange 'X@Y'  --data_format 'kg' --epoch 500
# done 

# # # for length task 
# for j in 1 2 3 4 5 6 7 8 9 10
# do
# echo 'length'
# python -u  gammaILP/code/main.py --target_predicate 'length' --task 'length' --number_variable 4 --target_variable_arrange 'X@Y' --minimal_precision 0.1 --random_negative 100 --data_format 'kg'
# done 

# # for uncle task and father task 
# for j in 1 2 3 4 5 6 7 8 9 10
# do
# echo 'uncle'
# python -u  gammaILP/code/main.py --target_predicate 'uncle' --task 'uncle' --number_variable 3 --target_variable_arrange 'X@Y'  --data_format 'kg'  --batch_size 256
# done 

# for j in 1 2 3 4 5 6 7 8 9 10
# do
# echo 'husband'
# python -u  gammaILP/code/main.py --target_predicate 'husband' --task 'husband' --number_variable 3 --target_variable_arrange 'X@Y'  --data_format 'kg'  --batch_size 256
# done 

#for fizz and buzz task
# for j in 1 2 3 4 5 6 7 8 9 10
for j in 1 
do
echo 'fizz'
python -u  gammaILP/code/main.py --target_predicate 'fizz' --task 'fizz' --number_variable 4 --target_variable_arrange 'X@X' --minimal_precision 1 --random_negative 200 --epoch 200 --lr_rule 0.01 --data_format 'kg' --device 'cuda:0'
done 

# python -u  gammaILP/code/main.py --target_predicate 'fizz' --task 'fizz' --number_variable 4 --target_variable_arrange 'X@X' --minimal_precision 1 --random_negative 100 --epoch 200 --lr_rule 0.1

# python -u  gammaILP/code/main.py --target_predicate 'fizz' --task 'fizz' --number_variable 4 --target_variable_arrange 'X@X' --minimal_precision 1 --random_negative 100 --epoch 200 --lr_rule 0.5

# for j in 1 2 3 4 5 6 7 8 9 10
for j in 1 
do
echo 'buzz'
python -u  gammaILP/code/main.py --target_predicate 'buzz' --task 'buzz' --number_variable 6 --target_variable_arrange 'X@X' --minimal_precision 1 --random_negative 200 --epoch 200 --lr_rule 0.01  --data_format 'kg' --device 'cuda:0'
done 

# python -u  gammaILP/code/main.py --target_predicate 'buzz' --task 'buzz' --number_variable 6 --target_variable_arrange 'X@X' --minimal_precision 1 --random_negative 100 --epoch 200 --lr_rule 0.1

# python -u  gammaILP/code/main.py --target_predicate 'buzz' --task 'buzz' --number_variable 6 --target_variable_arrange 'X@X' --minimal_precision 1 --random_negative 100 --epoch 200 --lr_rule 0.5





