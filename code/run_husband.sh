for j in 1 2 3 4 5 6 7 8 9 10
do
echo 'husband'
python -u  gammaILP/code/main.py --target_predicate 'husband' --task 'husband' --number_variable 2 --target_variable_arrange 'X@Y'  --data_format 'kg'  --batch_size 256 --epoch 20 --device 'cuda:0'
done 