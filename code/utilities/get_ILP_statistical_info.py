import os 
import sys 


# open folder and each line is 'e1 r e2', return the number of different entities and relations 
def get_entity_relation_number(father_path):
    entities = set()
    relations = set()
    with open(father_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('%'):
                parts = line.strip().split()
                if len(parts) == 3:
                    e1, r, e2 = parts
                    entities.add(e1)
                    entities.add(e2)
                    relations.add(r)
    return len(entities), len(relations)

def main(father_path):
    # open train.pl under all subfolders
    for root, dirs, files in os.walk(father_path):
        for file in files:
            if file == 'train.pl':
                file_path = os.path.join(root, file)
                entity_num, relation_num = get_entity_relation_number(file_path)
                print(f"File: {file_path}, Entities: {entity_num}, Relations: {relation_num}")
if __name__ == "__main__":
    father_path  = 'gammaILP/ILPdata'
    main(father_path)
    sys.exit(0)