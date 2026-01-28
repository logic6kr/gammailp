# repeat 5 times into a loop
for i in 1 2 3 4 5
do
    echo "Run $i times"
    python -u  gammaILP/code/baselines/cluster_ripper.py --target_predicate 'target' --task 'kandinsky_twopairs_50' --cluster_numbers 9 --target_variable_arrange 'X@X' --lr_rule 0.05 --lr_dkm 0.5 --batch_size 512 --epoch 200 --alpha 20 --lambda_dkm 4
done

for i in 1 2 3 4 5
do
    echo "Run $i times"
    python -u  gammaILP/code/baselines/cluster_ripper.py --target_predicate 'target' --task 'kandinsky_onered_4to1' --cluster_numbers 9 --target_variable_arrange 'X@X' --lr_rule 0.05 --lr_dkm 0.5 --batch_size 512 --epoch 200 --alpha 20 --lambda_dkm 4
done

for i in 1 2 3 4 5
do
    echo "Run $i times"
    python -u  gammaILP/code/baselines/cluster_ripper.py --target_predicate 'target' --task 'kandinsky_onetriangle_4to1' --cluster_numbers 9 --target_variable_arrange 'X@X' --lr_rule 0.05 --lr_dkm 0.5 --batch_size 512 --epoch 200 --alpha 20 --lambda_dkm 4
done
