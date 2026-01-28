import os 
# open the pl file 
def transfer(data_folder):
    all_entity = []
    all_relation = []
    all_triples = []
    with open(os.path.join(data_folder, 'train.pl'), 'r') as f:
        lines = f.readline()
        if lines[-1] == '\n':
            lines = lines[:-1]
        while lines:
            triples = lines.split(' ')
            first_entity = triples[0]
            relation = triples[1]
            second_entity = triples[2]
            if first_entity not in all_entity:
                all_entity.append(first_entity)
            if second_entity not in all_entity:
                all_entity.append(second_entity)
            if relation not in all_relation:
                all_relation.append(relation)
            all_triples.append((first_entity, relation, second_entity))
            lines = f.readline()
            if lines  and lines[-1] == '\n':
                lines = lines[:-1]
        f.close()
    # make entity file 
    with open(os.path.join(data_folder, 'entities.txt'), 'w') as f:
        for entity in all_entity:
            f.write(entity + '\n')
        f.close()
    # make relation file
    with open(os.path.join(data_folder, 'relations.txt'), 'w') as f:
        for relation in all_relation:
            f.write(relation + '\n')
        f.close()
    # make train file 
    with open(os.path.join(data_folder, 'train.txt'), 'w') as f:
        for triple in all_triples:
            f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
        f.close()
    # make test file
    with open(os.path.join(data_folder, 'test.txt'), 'w') as f:
        for triple in all_triples:
            f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
        f.close()
    # make valid file
    with open(os.path.join(data_folder, 'valid.txt'), 'w') as f:
        for triple in all_triples:
            f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
        f.close()
    # make all file 
    with open(os.path.join(data_folder, 'all.txt'), 'w') as f:
        for triple in all_triples:
            f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
        f.close()
    # make fact file 
    with open(os.path.join(data_folder, 'facts.txt'), 'w') as f:
        for triple in all_triples:
            f.write(triple[0] + '\t' + triple[1] + '\t' + triple[2] + '\n')
        f.close()

if __name__ == '__main__':
    data_folder = 'gammaILP/ILPdata/'
    task_name = 'adjr'
    data_folder = os.path.join(data_folder, task_name)
    transfer(data_folder)
    print(f"Data transfer completed for {data_folder}.")