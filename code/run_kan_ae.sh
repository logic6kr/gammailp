for i in 1 2 3 4 5
do
python -u  gammaILP/code/main.py --target_predicate 'target' --task 'kandinsky_onered' --cluster_numbers 8 --target_variable_arrange 'X@X' --lr_rule 0.05 --lr_dkm 0.5 --batch_size 512 --epoch 50 --alpha 20 --lambda_dkm 0.5 --pre_train_ae_num_epochs 100
done

for i in 1 2 3 4 5
do
python -u  gammaILP/code/main.py --target_predicate 'target' --task 'kandinsky_onetriangle' --cluster_numbers 8 --target_variable_arrange 'X@X' --lr_rule 0.05 --lr_dkm 0.5 --batch_size 512 --epoch 50 --alpha 20 --lambda_dkm 0.5 --pre_train_ae_num_epochs 100
done

for i in 1 2 3 4 5
do
python -u  gammaILP/code/main.py --target_predicate 'target' --task 'kandinsky_twopairs_50' --cluster_numbers 8 --target_variable_arrange 'X@X' --lr_rule 0.05 --lr_dkm 0.5 --batch_size 512 --epoch 50 --alpha 20 --lambda_dkm 0.5 --pre_train_ae_num_epochs 100
done