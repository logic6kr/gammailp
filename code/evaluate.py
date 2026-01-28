import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG)
import os
import sys
import itertools
import datetime
import pickle
from pyDatalog import pyDatalog 

class Evaluate:
    def __init__(self, task_name, target_predicate, test_file):
        self.First_call = True
        self.last_relation = ''
        self.left_bracket = '['
        self.right_bracket = ']'
        self.splitor = '@'
        self.target_predicate = target_predicate

    def return_all_predicate(self, original_data_path):
        '''
        Return all relations in the task 
        '''
        all_predicate = set([])
        with open(original_data_path,'r') as f:
                new_single_line  = f.readline()        
                while new_single_line:
                    # retrieval the relation name and two objects
                    predicate = new_single_line[0:new_single_line.index(self.right_bracket)]
                    single_line = predicate.split(self.left_bracket)
                    relation_name = single_line[0]
                    all_predicate.add(relation_name)
                    new_single_line = f.readline()
                f.close()
        return all_predicate


    def make_relation_entities(self, file_path):
        '''
        Usage: Reading .nl file and generate all symbolic predicate in a list or a dictionary. 
        The format of the generated dictionary is: {[relation] : [(pair_of_entities),...,(pair_of_entities)]}
        '''
        # Make relational_entitile 
        all_symbolic_predicate = []
        all_predicate = []
        relation = {}
        all_entities = set()
        logging.info('[Read Data] from:')
        logging.info(file_path)
        with open(file_path,'r') as f:
            single_line  = f.readline()        
            while single_line:
                one_perd = []
                # retrieval the relation name and two objects
                single_line = single_line[0:single_line.index(self.right_bracket)]
                single_line = single_line.split(self.left_bracket)
                relation_name = single_line[0]
                the_rest = single_line[1].split(self.splitor)
                first_obj = the_rest[0]
                second_obj = the_rest[1]
                if first_obj not in all_entities:
                    all_entities.add(first_obj)
                if second_obj not in all_entities:
                    all_entities.add(second_obj)
                one_perd.append(relation_name)
                one_perd.append(first_obj)
                one_perd.append(second_obj)
                all_predicate.append(one_perd)
                relation[relation_name] = []
                all_symbolic_predicate.append(relation_name+self.left_bracket+first_obj+self.splitor+second_obj+self.right_bracket)
                single_line = f.readline()
            f.close()
        
        for pred in all_predicate:
            one_tuple = []
            one_tuple.append(pred[1])
            one_tuple.append(pred[2])
            one_tuple = tuple(one_tuple) 
            relation[pred[0]].append(one_tuple) # {'relation_name': (),(),...,()}
        return relation, all_symbolic_predicate, list(all_entities)
        

    def make_deduction_in_mrr_hit_mode(self, task_name,data_path, t_relation, 
    result_path, logic_program_name, all_relation, hit_test_predicate = {}):
        '''
        - Check the MRR and HITS indicator with corrupted_pairs
        - No_training_mode: When open, only create database one time, else, create database every time when checking accuracy of logic programs. 
        - hit_test_predicate: A dictionary with 
        '''

        if self.last_relation != t_relation:
            generate_search = True
            self.last_relation = t_relation
        else:
            generate_search = False

        logic_path = result_path + '/' + logic_program_name        
        if not os.path.exists(logic_path):
            return -1
        precision_rule = [] 
        with open(logic_path, 'r') as f:
            rule = f.readline()[:-1]    
            body = []
            while rule:
                if self.splitor not in rule:
                    rule = f.readline()[:-1]
                    continue
                if '(' in rule:
                    latter_part = rule[rule.index('('):]
                    rule = rule[:rule.index('(')]
                    probability = float(latter_part[latter_part.index('(')+1:latter_part.index(',')])
                    precision_rule.append(probability)
                one_body = []
                rule = rule.replace(' ','')
                rule = rule.replace('-','')
                rule = rule.split(':')
                

                body_relation = rule[0]
                
                single_body = body_relation.split('^')
                for item in single_body:
                    one_body.append(item)

                body.append(one_body)
                rule = f.readline()[:-1]
            f.close()
            
        # ! build the variable into Datalog
        term_log = ''
        term_log += 'X,Y,Z,W,M,N,T,'
        for i in all_relation:
            term_log += i + ','
            
        term_log = term_log[0:-1]
        pyDatalog.create_terms(term_log)
        
        # Build database

        if self.First_call == True:
            self.create_database(task_name, True, True, False, data_path, t_relation, all_relation)
            self.First_call = False


        if generate_search == True:
            expression_list_all,self.variable_index = self.assemble_rule(body, all_relation, logic_path)
            search_index = 0 
            self.expression_list = []
            # for res in expression_list_all:
            #     x_index = variable_index[search_index]['X']
            #     y_index = variable_index[search_index]['Y']
            #     all_sub = np.array(res)
            #     try:
            #         two_entity = all_sub[:,[x_index, y_index]]
            #     except:
            #         two_entity = []
            #     cet = set(map(tuple, two_entity))
            #     expression_list.append(cet)
            #     search_index += 1 
            for res in expression_list_all:
                x_index = self.variable_index[search_index]['X']
                y_index = self.variable_index[search_index]['Y']
                try:
                    cet = set(map(lambda obj:(obj[x_index],obj[y_index]) , res))
                except:
                    cet = set([])
                self.expression_list.append(cet)
                search_index += 1 

        # ! each expression corresponds a rule
        # read the target predicate file 


        target_dic_set = set(hit_test_predicate)
        # The following check process follows the T_P operator of the logic program 

        research_index = 0 
        all_over_lap = set([])
        for res in self.expression_list: 
            # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
            # num_validate = 0
            # correct = 0
            # Get the minus set and return the value to the input dic 
            if len(res) == 0:
                research_index += 1
                continue
            overlap = res.intersection(target_dic_set)
            for ele in overlap:
                if isinstance(hit_test_predicate[ele][-1], (int, float)) and hit_test_predicate[ele][-1] <= precision_rule[research_index]:
                    hit_test_predicate[ele].append(precision_rule[research_index])
                else:
                    hit_test_predicate[ele].append(precision_rule[research_index])
                    
            all_over_lap = all_over_lap | overlap
            research_index += 1 
            
            # for re in res:
            #     # x_index = variable_index[search_index]['X']
            #     # y_index = variable_index[search_index]['Y']
            #     if x_index >= len(re) or y_index >= len(re):
            #         break
            #     num_validate += 1

        return hit_test_predicate, all_over_lap


    def create_database(self, task_name, test_mode, cap_flag, sample_walk, data_path, t_relation, all_relation):
        pyDatalog.clear()
        # ! build the variable into Datalog
        term_log = ''
        term_log += 'X,Y,Z,W,M,N,T,'
        for i in all_relation:
            term_log += i + ','
            
        term_log = term_log[0:-1]
        pyDatalog.create_terms(term_log)
        
        
        # # ! Build the databse 
        # if cap_flag == True:
        #     if sample_walk == True:
        #         all_trituple_path = data_path +task_name +'.onl'
        #     else:
        #         all_trituple_path = data_path +task_name +'.nl'
        # else:
        #     if sample_walk == True:
        #         all_trituple_path = data_path +t_relation+'.onl'
        #     else:
        #         all_trituple_path = data_path +t_relation+'.nl'
        
        # logging.info("Check the file with databae:")
        # logging.info(all_trituple_path)
        
        check_path = [data_path, data_path+'test']
        for data_path_single in check_path:
            with open(data_path_single, 'r') as f:
                single_tri = f.readline()
                while single_tri:
                    # skip the negative examples in the datasets
                    # if  '@' in single_tri and '-' not in single_tri:
                    if '-' in single_tri: # Do not build the negative examples
                        single_tri = f.readline()
                        continue
                    # Do not add any test data in the file
                    # !During the training process, do not inclue the testing facts. 
                    # !But during the testing process, containing the testing facts.
                    if test_mode == False and 'TEST' in single_tri:
                        single_tri = f.readline()
                        print('Skip Test Data')
                        continue
                    #prepare objects
                    single_tri = single_tri[:single_tri.index('#')]
                    relation_name = single_tri[:single_tri.index(self.left_bracket)]
                    first_entity = single_tri[single_tri.index(self.left_bracket) + 1 : single_tri.index(self.splitor)]
                    second_entity = single_tri[single_tri.index(self.splitor)+1 : single_tri.index(self.right_bracket)]
                    #add to database
                    + locals()[relation_name](first_entity,second_entity)
                    
                    single_tri = f.readline()
                f.close()
        return 0


    def assemble_rule(self, body,all_relation,logic_path):
        logging.info('Read logic program to Datalog from:')
        logging.info(logic_path)
        # ! build the variable into Datalog
        term_log = ''
        term_log += 'X,Y,Z,W,M,N,T,'
        for i in all_relation:
            term_log += i + ','
            
        term_log = term_log[0:-1]
        pyDatalog.create_terms(term_log)
        
        # Check each generated rules 
        expresison_list = []
        variable_index = []
        for rules in body:
            rule_skip = False
            
            flag = 0
            for item in rules:
                item = item[1:-1]
                negation_flag = False
                if item == '':
                    continue
                else:
                    name = item[:item.index(self.left_bracket)]
                    # if the neagation operator in the rule 
                    if '~' in name:
                        negation_flag = True
                        name = name[1:]
                    first_variable = item[item.index(self.left_bracket)+1: item.index(self.splitor)].upper()
                    second_variable = item[item.index(self.splitor)+1: item.index(self.right_bracket)].upper()
                    if flag == 0:
                        if negation_flag == True:
                            expression =  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                        else:
                            try:
                                expression =  locals()[name](locals()[first_variable],locals()[second_variable])
                            except:
                                rule_skip = True
                                break
                    else:
                        if negation_flag == True:
                            expression &=  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                        else:
                            try:
                                expression &=  locals()[name](locals()[first_variable],locals()[second_variable])
                            except:
                                rule_skip = True
                                break
                    flag += 1
            
            if rule_skip == True:
                logging.warning("Pass current rule")
                logging.warning(rules)
                continue
            
            # Find the order of variables x and y and z
            str = ''
            va_f = 0
            for i in rules:
                str += i
            o_variable_index = []
            for var in ['X','Y','Z','W','M','N','T']:
                if var in str:
                    index = str.index(var)
                    va_f = 1
                else:
                    index = 1e6
                o_variable_index.append((var, index))
            
            o_variable_index.sort(key= lambda y:y[1])
            var_dic = {}
            for i in range(len(o_variable_index)):
                var_dic[o_variable_index[i][0]] = i
            if va_f == 1:
                variable_index.append(var_dic)
            if flag != 0:
                expresison_list.append(expression)
        
        
        logging.info('Create new logic program to Datalog success.')
        return expresison_list, variable_index
        



    def check_MRR_Hits(self, task_name, target_predicate, test_file):
        MRR_mean = []
        hits_number = [1,3,10]
        hits_info = [[] for i in range(len(hits_number))]
        
        data_path = f'gammaILP/cache/{task_name}/{target_predicate}.nl'
        result_path = f'gammaILP/cache/{task_name}/'
        
        # if not os.path.exists(f'gammaILP/cache/{task_name}/{target_predicate}.nl'):
        #     shutil.copy(f'gammaILP/cache/{task_name}/{target_predicate}.nl', f'gammaILP/cache/{task_name}/{target_predicate}.onl')
        
        test_file_path = f'gammaILP/cache/{task_name}/{target_predicate}.nltest'
        
        all_relation = self.return_all_predicate(data_path)


        test_r2e,_,_ = self.make_relation_entities(test_file_path)
        _,all_symbolic_predicate,all_ent = self.make_relation_entities(data_path)
        
        # update all predicate and all ent

        all_symbolic_predicate_set = set(all_symbolic_predicate)
        def make_all_test_atoms(relation, start_flag, max_single_scanned):
            '''
            Make all corrupt atoms according each test atoms with a same relation.
            '''
            logging.info('Make all corrupt atoms for the relation:')
            logging.info(relation)
            all_test_atom = {}
            pairs_rank = {}

            # this_trun_start_flag = start_flag
            if start_flag >= len(test_r2e[relation]):
                return -1,-1,-1
            if start_flag + max_single_scanned >= len(test_r2e[relation]):
                end_flag = len(test_r2e[relation])
            else:
                end_flag  = start_flag + max_single_scanned 
            for pairs in test_r2e[relation][start_flag:end_flag]:
                # for each test facts
                all_test_atom[pairs] = []
                for kept_entity_index in range(2):
                    new_pair = (pairs[0], pairs[1],kept_entity_index )
                    pairs_rank [new_pair]  = []
                    change_entity_index = (kept_entity_index+1) % 2
                    kept_entity = pairs[kept_entity_index]
                    if change_entity_index == 1:
                        corrupted_pairs = list(itertools.product([kept_entity],all_ent))
                    else:
                        corrupted_pairs = list(itertools.product(all_ent,[kept_entity]))
                    for i in corrupted_pairs:
                        single = relation + self.left_bracket+i[0]+self.splitor+i[1] + self.right_bracket
                        if single in all_symbolic_predicate_set:
                            continue
                        if i not in all_test_atom:
                            all_test_atom[i] = []
                            all_test_atom[i].append(new_pair)
                        else:
                            all_test_atom[i].append(new_pair)
                    all_test_atom[pairs].append(new_pair)
                
                
            logging.info('Make corupted atoms success!')
            return all_test_atom, pairs_rank, start_flag+max_single_scanned

        total_test = 0
        for relation in test_r2e:
            logging.info("[MRR HITS]Check relation on:")
            logging.info(relation)
            
            start_flag = 0
            maximum_fact_single = 200
            while True:
                all_test_atom, pairs_rank, start_flag  = make_all_test_atoms(relation, start_flag, maximum_fact_single)
                if all_test_atom == -1:
                    break
                
                all_test_atom, overlap = self.make_deduction_in_mrr_hit_mode(task_name, data_path, relation, result_path, test_file, all_relation, hit_test_predicate = all_test_atom)
                
                for i in overlap:
                    for p in all_test_atom[i]:
                        if not isinstance(p, (int, float)):
                            pairs_rank[p].append((i, all_test_atom[i][-1]))
                
                logging.info("Begin computing for the test instance")
                for target_fact in pairs_rank:
                    # logging.info(target_fact)
                    test_pro = pairs_rank[target_fact]
                    test_pro.sort(key = lambda x: x[1], reverse = True)

                    the_last_pro = 1e8
                    correct_rank = 1e8
                    tem_correct_rank = 0
                    for all_corupt_pro in test_pro:
                        if all_corupt_pro[1] < the_last_pro:
                            tem_correct_rank += 1
                            the_last_pro = all_corupt_pro[1]
                        if all_corupt_pro[0] == (target_fact[0], target_fact[1]):
                            correct_rank = tem_correct_rank
                            break
                    
                    
                    MRR_mean.append(1/correct_rank)
                    
                    total_test += 1
                    
                    hit_index = 0
                    for i in hits_number:
                        if correct_rank <= i:
                            hits_info[hit_index].append(relation+self.left_bracket+target_fact[0]+self.splitor+target_fact[1]+self.right_bracket)
                        hit_index += 1
                        
                    with open(result_path+'MRR'+test_file+'.new.dt','wb') as f:
                        pickle.dump(MRR_mean,f)
                        f.close()
                    with open(result_path+'HINT'+test_file+'.new.dt','wb') as f:
                        pickle.dump(hits_info,f)
                        f.close()
                        
                    MRR_mean_value = sum(MRR_mean)/ len(MRR_mean)
                    
                    hit_value = []
                    for i in hits_info:
                        a = len(i)
                        hit_value.append(a/total_test)
                        
                with open(result_path+'MRR'+test_file+'.new.txt','a') as f:
                    print(datetime.datetime.now(), file = f)
                    print(MRR_mean, file=f)
                    print(MRR_mean_value, file=f)
                    print('instance have checked:',total_test/2, file=f)
                    f.close()
                    
                with open(result_path+'HINT'+test_file+'.new.txt','a') as f:
                    print(datetime.datetime.now(),file = f)
                    print(hits_info, file=f)
                    print(hit_value, file=f)
                    print('instance have checked:',total_test/2, file=f)
                    f.close()
                    
        
        logging.info('Check [MRR and HINTS] for [%s] Success!'%task_name)
        logging.info('MRR')
        logging.info(MRR_mean_value)
        logging.info('HITNS@[1,3,10]')
        logging.info(hit_value)
        return MRR_mean_value, hit_value

if __name__ == '__main__':
    task_name = sys.argv[1] if len(sys.argv) > 1 else 'mnist'
    target_predicate = sys.argv[2] if len(sys.argv) > 2 else 'lessthan'
    test_file = f'rule_output_{target_predicate}_chain_substitution_no_neural_predicate.md'
    evaluator = Evaluate(task_name, target_predicate, test_file)
    evaluator.check_MRR_Hits(task_name, target_predicate, test_file)
    
    # task_name = 'WN18RR'
    # target_predicate = 'P1001'
    # test_file = 'test'
    # sample_walk = False
    # check_MRR_Hits(task_name, target_predicate, test_file, sample_walk)