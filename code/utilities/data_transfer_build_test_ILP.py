import os 
import sys
# open the pl file 
def transfer(data_folder, output_dir, subset = 'train'):
    all_triples = []
    with open(os.path.join(data_folder, subset), 'r') as f:
        lines = f.readline()
        while lines:
            if lines[-1] == '\n':
                lines = lines[:-1]
            triples = lines.split('\t')
            first_entity = triples[1]
            relation = triples[0]
            second_entity = triples[2]
            all_triples.append((first_entity, relation, second_entity))
            lines = f.readline()
        f.close()
    # make train file
    with open(os.path.join(output_dir, f'{subset}.pl'), 'w') as f:
        for triple in all_triples:
            f.write(triple[0] + ' ' + triple[1] + ' ' + triple[2] + '\n')
        f.close()
    # make test file
def build_all(data_folder, output_dir):
    all_triples = []
    with open(os.path.join(data_folder, 'train'), 'r') as f:
        lines = f.readline()
        while lines:
            if lines[-1] == '\n':
                lines = lines[:-1]
            triples = lines.split('\t')
            first_entity = triples[1]
            relation = triples[0]
            second_entity = triples[2]
            all_triples.append((first_entity, relation, second_entity))
            lines = f.readline()
        f.close()
    with open(os.path.join(data_folder, 'test'), 'r') as f:
        lines = f.readline()
        while lines:
            if lines[-1] == '\n':
                lines = lines[:-1]
            triples = lines.split('\t')
            first_entity = triples[1]
            relation = triples[0]
            second_entity = triples[2]
            all_triples.append((first_entity, relation, second_entity,'TEST'))
            lines = f.readline()
        f.close()
    # make all file
    with open(os.path.join(output_dir, 'all.pl'), 'w') as f:
        for triple in all_triples:
            if len(triple) == 4:
                f.write(triple[0] + ' ' + triple[1] + ' ' + triple[2] + ' ' + triple[3] + '\n')
            else:
                f.write(triple[0] + ' ' + triple[1] + ' ' + triple[2] + '\n')
        f.close()

if __name__ == '__main__':
    data_folder = 'gammaILP/ILPdata/'
    # task_name is sys first argument, can be 'uncle' or 'husband'
    task_name = sys.argv[1] if len(sys.argv) > 1 else 'uncle'
    # task_name = 'husband'
    output_dir = 'gammaILP/ILPdata/'
    output_dir = os.path.join(output_dir, task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_folder = os.path.join(data_folder, task_name)
    transfer(data_folder, output_dir, subset='train')
    transfer(data_folder, output_dir, subset='test')
    build_all(data_folder, output_dir)
    print(f"Data transfer completed for {data_folder}.")