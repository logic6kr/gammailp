import sys
import pickle
from pyDatalog import pyDatalog
from pydantic import BaseModel

class Step(BaseModel):
    explanation: str
    output: str

class Rules(BaseModel): 
    head: str
    body: str
    
class LogicProgram(BaseModel):
    steps: list[Step]
    final_logic_program: list[Rules]


def test_chat(father_path='LLM_induce_logic', file_name='test'):
    with open(f"{father_path}/res/{file_name}.md", "r") as f:
        ans = f.read()
        f.close()
    return ans


def  build_target_predicate(t_relation, task_name, test_mode=False, cap_flag = False, data_path = 'gammaILP/ILPdata/', result_path = 'gammaILP/cache/'):
    '''
    build target predicate and return a dictionary consisting the predicate 
    '''
    data_path = data_path + task_name 
    result_path = result_path + task_name + '/'
    all_target_predicate_dic = {}
    if cap_flag == True:
        facts_path = data_path+'/all.pl'
    else:
        facts_path = data_path+'/all.pl'
        
    all_facts = []
    with open(facts_path, 'r') as f:
        single_line = f.readline()
        while single_line:
            # skip the negative examples in the datasets, we won't add the negative examples in test datasets
            if '-' in single_line:
                single_line = f.readline()
                continue
            if test_mode == True:
                # if the test mode is True, then check the accuracy based on the test datasets 
                all_facts.append(single_line.replace('\n',''))
                if 'TEST' in single_line:
                    head_relation = single_line.split(' ')[1]
                    if head_relation == t_relation:
                        all_target_predicate_dic[single_line[:single_line.index('TEST')-1]] = 0
            else:
                # if the test mode is Off, then check auucracy expecting the test datasets
                if 'TEST' in single_line:
                    single_line = f.readline()
                    continue
                head_relation = single_line.split(' ')[1]
                all_facts.append(single_line.replace('\n',''))
                if head_relation == t_relation:
                    all_target_predicate_dic[single_line.replace('\n','')] = 0
            single_line = f.readline()
        f.close()
    with open(result_path+'target_pred.txt','w') as f:
        print(all_target_predicate_dic, file=f)
        f.close()
    with open(result_path+'target_pred.dt','wb') as f:
        pickle.dump(all_target_predicate_dic, file=f)
        f.close()
        
    #  build all relations
    all_relations = []
    with open(facts_path, 'r') as f:
        single_line = f.readline()
        while single_line:
            if '-' in single_line:
                single_line = f.readline()
                continue
            head_relation = single_line.split(' ')[1]
            if head_relation not in all_relations:
                all_relations.append(head_relation)
            single_line = f.readline()
        f.close()
    
    return all_target_predicate_dic, all_relations, all_facts

def  read_target_predicate(task_name, t_relation):
    '''
    Read target predicate and return a dictionary consisting the predicate 
    '''
    result_path = 'DFOL/' + task_name + '/data/' + t_relation+'/'
    with open(result_path+'target_pred.dt','rb') as f:
        dic = pickle.load(f)
        f.close()
    return dic


def check_accuracy_of_logic_program(task_name,logic_rules, t_relation, hit_flag = False,t_arity=1, test_mode = False, hit_test_predicate = [], cap_flag = False):
    '''
    Input the symbolic logic program, return the correctness of each rule in the logic program.
    '''
    all_target_predicate_dic, all_relation, all_facts = build_target_predicate(t_relation, task_name, test_mode, test_mode)
    body = []
    precision_rule = [] 
    head = []
    for rule in logic_rules:
        # if no @ continue
        if '@' not in rule:
            continue
        if '(' in rule:
            # if the rule contains the probability, then extract the probability the probabilit is. in [father[X@Z] == 1] ^ [father[Z@Y] == 1] (1.0, 3, 3) 1. 
            latter_part = rule[rule.index('('):]
            probability = float(latter_part[latter_part.index('(')+1:latter_part.index(',')])
            precision_rule.append(probability)
        one_body = []
        rule = rule.replace(' ','')
        rule_body = rule[:rule.index('(')]
        
        head_relation = t_relation
        try:
            body_relation = rule_body
            single_body = body_relation.split('^')     
            body.append(single_body)
        except:
            body.append([])
        

    # ! build the variable into Datalog
    term_log = ''
    term_log += 'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,'
    all_variable_list = term_log[0:-1].split(',')
    for i in all_relation:
        term_log += i + ','
        
    term_log = term_log[0:-1]
    pyDatalog.create_terms(term_log)
    
    # ! Build the database 
    for single_tri in all_facts:
        # skip the negative examples in the datasets
        # if  '@' in single_tri and '-' not in single_tri:
        if '-' in single_tri: # Do not build the negative examples
            continue
        # Do not add any test data in the file
        # !During the training process, do not inclue the testing facts. 
        # !But during the testing process, containing the testing facts.
        if test_mode == False and 'TEST' in single_tri:
            continue
        #prepare objects
        relation_name = single_tri.split(' ')[1]
        first_entity = single_tri.split(' ')[0]
        second_entity = single_tri.split(' ')[2]
        # add unary relation to db 
        if first_entity == second_entity:
            + locals()[relation_name](first_entity)
        else:
            #add to database
            + locals()[relation_name](first_entity,second_entity)
    
    # Check each generated rules 
    expresison_list = []
    variable_index = []
    for rules in body:
        # Find the order of variables x and y and z
        str = ''
        va_f = 0
        for i in rules:
            str += i
        o_variable_index = []
        for var in all_variable_list:
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
        flag = 0
        for item in rules:
            negation_flag = False
            item = item[1:-1]
            if item == '':
                continue
            else:
                name = item[:item.index('[')]
                # if the negation operator in the rule 
                if '~' in name:
                    negation_flag = True
                    name = name[1:]
                if '@' not in item:
                    first_variable = item[item.index('(')+1: item.index(')')].upper()
                    if flag == 0:
                        if negation_flag == True:
                            expression =  ~(locals()[name](locals()[first_variable]))
                        else:
                            expression =  locals()[name](locals()[first_variable])
                    else:
                        if negation_flag == True:
                            expression &=  ~(locals()[name](locals()[first_variable]))
                        else:
                            expression &=  locals()[name](locals()[first_variable])
                else:
                    first_variable = item[item.index('[')+1: item.index('@')].upper()
                    second_variable = item[item.index('@')+1: item.index(']')].upper()
                    if flag == 0:
                        if negation_flag == True:
                            expression =  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                        else:
                            expression =  locals()[name](locals()[first_variable],locals()[second_variable])
                    else:
                        if negation_flag == True:
                            expression &=  ~(locals()[name](locals()[first_variable],locals()[second_variable]))
                        else:
                            expression &=  locals()[name](locals()[first_variable],locals()[second_variable])
                flag += 1
        if flag != 0:
            expresison_list.append(expression)
            
    # ! each expression corresponds a rule
    # read the target predicate file 
    if hit_flag == True:
        target_dic = hit_test_predicate
    else:
        target_dic = all_target_predicate_dic
    
    # The following check process follows the T_P operator of the logic program 
    correct_f = []
    search_index = 0 
    for res in expresison_list: # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
        num_validate = 0
        correct = 0
        head_relation = t_relation
        if t_arity == 1:
            first_head = 'X'
            second_head = first_head
            t_arity = 1
        else:
            first_head = 'X'
            second_head = 'Y'
            t_arity = 2
        try:
            for re in res:
                x_index = variable_index[search_index][first_head]
                y_index = variable_index[search_index][second_head]
                if x_index >= len(re) or y_index >= len(re):
                    break
                num_validate += 1
                if t_arity == 2:
                    first_res = re[x_index]
                    second_res = re[y_index]
                    final = len(locals()[head_relation](first_res,second_res)) # The ground predicate 
                elif t_arity == 1:
                    first_res = re[x_index]
                    second_res = re[x_index]
                    final = len(locals()[head_relation](first_res)) # The ground predicate
                # ! test mode open iff when check accuracy basedon the .nl file
                if test_mode == True:
                    predicate =  first_res+' '+head_relation+' '+second_res
                    if predicate in target_dic:
                        if hit_flag == True:
                            current_precision_value = target_dic[predicate]
                            # iff the new precision are larger than the current one, updata the value 
                            if precision_rule[search_index] >= current_precision_value: 
                                target_dic[predicate] = precision_rule[search_index]
                        else:
                            target_dic[predicate] = 1
                        correct += 1    
                # ! when the test mode is false and the target ground predicate in the database
                elif final == 1:  
                    if t_arity == 1:
                        predicate = first_res+ ' ' + head_relation + ' ' + first_res   
                    elif t_arity == 2:
                        predicate = first_res+' ' +head_relation + ' '+second_res
                    if predicate in target_dic:
                        target_dic[predicate] = 1
                    correct += 1
            if num_validate == 0:
                num_validate = -1
            correct_f.append((correct/num_validate ,correct, num_validate))
        except:
            correct_f.append((0,-1,-1))
        search_index += 1
    print('The accuracy of rules:', correct_f)
    
    if hit_flag == True:
        correct_f = target_dic
    
    print('The target examples:', target_dic)
    
    cover_rate = 0 
    for item in target_dic:
        if target_dic[item] == 1:
            cover_rate += 1
    cover_rate = cover_rate/len(target_dic)
    
    return correct_f, target_dic, cover_rate    

def find_rules(task_name):
    with open(task_name, 'r') as f:
        gpt_output = f.read()
        f.close()
    each_line = gpt_output.split('\n')
    rules = []
    for item in each_line:
        try:
            if '@' in item:
                valid_rule = item[:-1]
                rules.append(valid_rule)
        except:
            continue
    return rules
    

def main(target_predicate = 'uncle', task_name = 'uncle'):
    rule_path=f'gammaILP/cache/{target_predicate}/llm.md'
    logic_rules = find_rules(rule_path)
    accuracy, target_predicate_values, cover_rate = check_accuracy_of_logic_program(task_name, logic_rules, target_predicate, test_mode = True, t_arity=2)
    print(f"Precision of logic rules: {accuracy}")
    print(f"Recall: {cover_rate}")
    
if __name__ == '__main__':
    target_predicate = sys.argv[1] if len(sys.argv) > 1 else 'related'
    task_name = sys.argv[2] if len(sys.argv) > 2 else 'related'
    main(target_predicate=target_predicate, task_name=task_name)