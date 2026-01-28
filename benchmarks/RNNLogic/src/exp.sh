for i in 'adjr' 'buzz' 'cons' ''even' 'even20' 'father' 'fizz' 'gcor' 'gp' 'icews' 'iscycle' 'lessthan' 'member' 'odd' 'pre' 'related' 'son' 'tc' 'ue' 
do
    cd ../miner/
    ./rnnlogic -data-path ../data/${i} -max-length 3 -threads 40 -lr 0.01 -wd 0.0005 -temp 100 -iterations 1 -top-n 0 -top-k 0 -top-n-out 0 -output-file ../data/${i}/mined_rules.txt
    
done