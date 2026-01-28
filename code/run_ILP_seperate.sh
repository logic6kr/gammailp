for i in 'cons' 'father' 'lessthan'
do
for j in 1 2 3 4 5 6 7 8 9 10
do
    echo ${i}
    python -u  gammaILP/code/main.py --target_predicate ${i} --task ${i} --number_variable 3 --target_variable_arrange 'X@Y'  --data_format 'kg'
done
done
