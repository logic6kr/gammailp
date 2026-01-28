import sys, os, time
import sklearn.metrics
from datetime import datetime
sys.path.append('gammaILP/')
sys.path.append('gammaILP/logic_back/')
from torch.utils.tensorboard import SummaryWriter
# from rule_learning_original.code.pykan.kan import *
## set the argument
import itertools 
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import collections
import torch.utils 
from aix360_k.aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360_k.aix360.algorithms.rule_induction.trxf.core.conjunction import Conjunction
from aix360_k.aix360.algorithms.rule_induction.trxf.core.feature import Feature
from aix360_k.aix360.algorithms.rule_induction.trxf.core.predicate import Predicate, Relation
import aix360_k.aix360.algorithms.rule_induction.trxf.classifier.ruleset_classifier as trxf_classifier
from pyDatalog import pyDatalog 
import pickle
from collections import deque



class CoreLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    
    def __init__(self, in_size, rule_number, default_alpha = 10, device='cuda:0',output_rule_file = '', t_arity=2, time_stamp=''):
        super().__init__()
        self.size_in = in_size
        self.rule_number = rule_number
        self.alpha = default_alpha
        self.device = device
        self.output_rule_file = output_rule_file
        self.good_rule_embeddings = None
        self.current_final_interpret = None
        self.time_stamp = time_stamp

        try:
            with open(f'{self.output_rule_file}weights{self.time_stamp}.pkl', 'rb') as f:
                self.good_rule_embeddings = torch.from_numpy(pickle.load(f)).to(device)
                f.close()
            print('Retrial good rule embeddings')
        except:
            print('No good rule embeddings')
        self.t_arity = t_arity

    # def sigmoid_like(self, x):
    #     y = torch.divide(1,(1+torch.exp(-1 * self.alpha *(x))))  
    #     return y 
    
    # def fuzzy_or(self, x):   
    #     neg_inputs = 1 - x
    #     pro_neg_input = torch.prod(neg_inputs, 1)
    #     predict_value = 1 - pro_neg_input
    #     predict_value = torch.reshape(predict_value, [-1, 1])
    #     return predict_value
    
    # def forward(self, x):
    #     self.interpretable_rul_weights = torch.nn.functional.softmax(self.weights, dim=1)
    #     w_times_x= torch.mm(x, self.interpretable_rul_weights.t())
    #     biased = w_times_x - 0.5
    #     activated_values =  2 * nn.functional.leaky_relu(biased, negative_slope=0.1)
    #     rule_satisfy = self.fuzzy_or(activated_values)
    #     return rule_satisfy # let the rule satisfy be 1 always 
    
    def interpret_weights_computation(self):
        pass
        # '''
        # return the interpretable matrix based on the weights of the model 
        # '''
        # self.current_final_interpret = self.interpretable_rul_weights
        # return self.interpretable_rul_weights
    
    def similiarity(self):
        # enlarge the dissimilarity between the rows of weights 
        current_final_interpret = self.interpret_weights_computation()
        rule_dis_loss = 0
        all_combination_rules = list(itertools.combinations(range(0, self.rule_number), 2))
        for item in all_combination_rules:
            first_row = current_final_interpret[item[0],:]
            second_row = current_final_interpret[item[1],:]
            rule_dis_loss += torch.nn.functional.cosine_similarity(first_row, second_row, dim=0)
        return rule_dis_loss
    
    def similarity_with_learned_rules(self):
        try:
            current_final_interpret = self.interpret_weights_computation()
            rule_dis_loss = 0
            for item_1 in current_final_interpret:
                for item_2 in self.good_rule_embeddings:
                    first_row = item_1
                    second_row = item_2
                    rule_dis_loss += torch.nn.functional.cosine_similarity(first_row, second_row, dim=0)
            return rule_dis_loss
        except:
            return 0
    
    def connective_loss(self):
        # no connective loss for the rule layer on time series data 
        return 0 
    
    def interpret(self, x_train:pd.DataFrame, precision = 100, existing_rule_set  = None, scale = False):
        '''
        Input from pandas form data and output the symbolic rules.
        If there is old rule set, add them into the new rule set
        
        @param x_train: the input data columns or list with all unground body atoms
        @param precision: the precision of the threshold to get the rules
        '''
        current_final_interpret = self.interpret_weights_computation()
        if type(x_train) is pd.DataFrame:
            self.atoms = list(x_train.columns)
            try:
                self.atoms.remove('label')
            except:
                print('no label in the data')
        else:
            self.atoms = x_train
        string_predicates = set([])
        all_conjunctions = []
        interpretable_weights_cpu = current_final_interpret.cpu().detach().numpy()
        if scale  == True:
            # Add epsilon to avoid log(0)
            eps = 1e-6
            interpretable_weights_cpu = np.clip(interpretable_weights_cpu, eps, 1 - eps)
            logit = np.log(interpretable_weights_cpu / (1 - interpretable_weights_cpu))  # inverse sigmoid
            # Normalize back to 0-1 range
            interpretable_weights_cpu = (logit - logit.min()) / (logit.max() - logit.min())
        for threshold_step in range(1,precision):
            threshold = threshold_step/precision
            activated_index = np.where(interpretable_weights_cpu > threshold)
            pairs_rule_atoms = collections.defaultdict(list) # store the rule index and highlighted atoms [[item](rule_index)...[]]
            index_rules = list(activated_index[0])
            values_items = list(activated_index[1])

            for index, value in enumerate(index_rules):
                corresponding_atoms = self.atoms[values_items[index]]
                pairs_rule_atoms[value].append(corresponding_atoms)

            for key, item_list in pairs_rule_atoms.items():
                predicate_list = []
                for item in item_list:
                    predicate_list.append(Predicate(Feature(item), Relation.EQ, 1))
                conjunct = Conjunction(predicate_list)
                if str(conjunct) not in string_predicates:
                    string_predicates.add(str(conjunct))
                    all_conjunctions.append(conjunct)
        rules_obj = DnfRuleSet(all_conjunctions, 1)
        # add conjunctions into the existing rule set
        if type(existing_rule_set) != type(None):
            for conj in existing_rule_set.conjunctions:
                rules_obj.add_conjunction(conj)
            # print(rules_obj)
        return rules_obj
    

    def check_metric(self,test_data:pd.DataFrame=None,rule_set:DnfRuleSet=None, remove_low_precision=0.5,print_result=True,update_parameters=False,single_check=False, append_info=None): # type: ignore
        '''
        - When the data if in propositional format like time series or tabular, check the acc with this function
        - Copied from Time series class, need to set remove the low precision rules or not;
        - **Single check** indicate the model only learn a single rule and check the metrics for rules generated in a single running batch. In this mode, some invented predicates may be checked;
        '''
        if type(test_data) is pd.DataFrame:
            self.y_test = test_data['label']
            self.x_test = test_data.drop('label', axis = 1)
        
        classifier = trxf_classifier.RuleSetClassifier([rule_set],rule_selection_method=trxf_classifier.RuleSelectionMethod.FIRST_HIT,confidence_metric=trxf_classifier.ConfidenceMetric.LAPLACE,weight_metric=trxf_classifier.WeightMetric.CONFIDENCE,default_label=0)
        classifier.update_rules_with_metrics(self.x_test, self.y_test)
        number_rules = len(rule_set.conjunctions)
        removed_list = []
        if remove_low_precision != None:
            print('[Prune] remove low precision rules:')
            rule_removed = classifier.remove_low_precision(lower_threshold=remove_low_precision)
            rule_set = rule_removed['new_rule']
            removed_list = rule_removed['removed_rules']
            classifier = trxf_classifier.RuleSetClassifier([rule_set],rule_selection_method=trxf_classifier.RuleSelectionMethod.FIRST_HIT,confidence_metric=trxf_classifier.ConfidenceMetric.LAPLACE,weight_metric=trxf_classifier.WeightMetric.CONFIDENCE,default_label=0)
            classifier.update_rules_with_metrics(self.x_test, self.y_test)
            new_number_rules = len(rule_set.conjunctions)
            print(rule_set)
            print(f'Remove {number_rules-new_number_rules} Low Precision Rules')
            
        precision = []
        recall = []
        tp = []
        tn = []
        fp = []
        fn = []
        lift = []

        for rule in classifier.rules:
            precision.append(rule.precision)
            recall.append(rule.recall)
            tp.append(rule.tp)
            tn.append(rule.tn)
            fp.append(rule.fp)
            fn.append(rule.fn)
            lift.append(rule.lift)
        
        if 1 in precision and update_parameters == True:
            current_final_interpret = self.interpret_weights_computation().cpu().detach().numpy()
            try: 
                with open(f'{self.output_rule_file}weights{self.time_stamp}.pkl', 'rb') as f:
                    old_final_interpret = pickle.load(f)
                    f.close()
                new_final_interpret = np.concatenate((old_final_interpret, current_final_interpret), axis=0)
                with open(f'{self.output_rule_file}weights{self.time_stamp}.pkl', 'wb') as f:
                    pickle.dump(new_final_interpret, f)
                    f.close()
            except:
                with open(f'{self.output_rule_file}weights{self.time_stamp}.pkl', 'wb') as f:
                    pickle.dump(current_final_interpret, f)
                    f.close()
            
        all_rules = str(rule_set)
        rules_str = all_rules.replace('v','')
        rules_str = rules_str.split('\n')[1:-2]
        new_rule = []
        try:
            for index,item in enumerate(rules_str):
                new_rule.append(item + f'precision: {precision[index]} recall: {recall[index]} lift {lift[index]} tp: {tp[index]} tn: {tn[index]} fp: {fp[index]} fn: {fn[index]}' + '\n')
        except:
            print('No Rule in the Rule Set')
            metrics = {'precision':0, 'recall':0, 'lift':0 , 'rule_set_precision':0, 'rule_set_recall':0, 'acc':0}
            return_obj = {'metrics':metrics, 'rule_set':rule_set, 'classifier':None, 'removed_list':[]}
            return return_obj
        if print_result == True:
            with open(f'{self.output_rule_file}', 'a+') as f:
                time_str = datetime.today().strftime(f'%Y-%m-%d %H:%M:%S Single Check: {single_check}')
                print(f'**{time_str}**', file=f)
                print('Number of Layer:', 2, file=f)
                # print('Seed', self.seed, file=f)
                for i in new_rule:
                    print(i, file=f)
                f.close()
        
        # make prediction based on rule
        y_pred = [] 
        for item in range(len(self.x_test)):
            y_pred.append(classifier.predict(self.x_test.iloc[item]))
        y_test = self.y_test.to_numpy()
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        if print_result == True:
            with open(f'{self.output_rule_file}', 'a+') as f:
                print(f'tn: {tn} fp: {fp} fn: {fn} tp: {tp}', file=f)
                print(f'acc_test:{(tp+tn)/(tn+fp+fn+tp)}', f'precision:{tp/(tp+fp)}', f'recall:{tp/(tp+fn)}', f'f1:{2*tp/(2*tp+fp+fn)}, append_info: {append_info}', file=f)
                print(f'From Rule: acc_test:{(tp+tn)/(tn+fp+fn+tp)}', f'precision:{tp/(tp+fp)}', f'recall:{tp/(tp+fn)}', f'f1:{2*tp/(2*tp+fp+fn)}, append_info: {append_info}')
        rule_set_precision = tp/(tp+fp)
        rule_set_recall = tp/(tp+fn)
        rule_set_f1 = 2*tp/(2*tp+fp+fn)
        acc = (tp+tn)/(tn+fp+fn+tp)
        metrics = {'precision':precision, 'recall':recall, 'lift':lift , 'rule_set_precision':rule_set_precision, 'rule_set_recall':rule_set_recall, 'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'f1':rule_set_f1, 'acc': acc}
        return_obj = {'metrics':metrics, 'rule_set':rule_set, 'classifier':classifier, 'removed_list':removed_list}
        return return_obj

class AndLayer(nn.Module):
    '''
    The logical and layer for deep Rule 
    '''
    def __init__(self, number_features, number_aux_pred, default_alpha = 1):
        super(AndLayer, self).__init__()
        self.weights = nn.init.normal_(nn.Parameter(torch.Tensor(number_features, number_aux_pred), requires_grad=True), mean=0, std = 0.1)
        # self.dropout = torch.nn.Dropout(p=0.5)
        self.default_alpha = default_alpha
    def forward(self, x):
        # weight_dp = self.dropout(self.weights)
        self.interpretable_rul_weights = torch.nn.functional.softmax(self.weights, dim=0)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        w_times_x= torch.mm(x, self.interpretable_rul_weights)
        biased = w_times_x - 0.5
        # TODO the activateed function can be replaced into softmax and the loss function can be used as cross entropy correspondingly
        activated_aux_predicates = 2 * self.default_alpha * nn.functional.leaky_relu(biased, negative_slope=0.01)
        # activated_aux_predicates =  2 * nn.functional.relu(biased)
        # activated_aux_predicates =  nn.functional.relu(w_times_x)
        return activated_aux_predicates
    
    def interpret(self, atoms, threshold):
        interpretable_weights_cpu = self.weights.cpu().detach().numpy()
        activated_index = np.where(interpretable_weights_cpu > threshold)
        pairs_rule_atoms = collections.defaultdict(list) # store the rule index and highlighted atoms [[item](rule_index)...[]]
        index_rules = list(activated_index[0])
        values_items = list(activated_index[1])

        for index, value in enumerate(index_rules):
            corresponding_atoms = atoms[values_items[index]]
            pairs_rule_atoms[value].append(corresponding_atoms)
        
        return pairs_rule_atoms

class OrLayer(nn.Module):
    def __init__(self, ):
        super(OrLayer, self).__init__()
        
    def fuzzy_or(self, x):   
        neg_inputs = 1 - x
        pro_neg_input = torch.prod(neg_inputs, 1)
        predict_value = 1 - pro_neg_input
        predict_value = torch.reshape(predict_value, [-1, 1])
        return predict_value
    
    def forward(self, x):
        '''
        x is the activated aux predicates or head predicates
        '''
        rule_satisfy = self.fuzzy_or(x)
        return rule_satisfy # let the rule satisfy be 1 always 


class BaseRule(CoreLayer):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, in_size, rule_number, default_alpha = 10, device='cuda:0',output_rule_file = './a_rule.md', t_relation = '', task = '',t_arity=2,number_aux_pred=5, time_stamp=''):
        super().__init__(in_size, rule_number, default_alpha, device, output_rule_file,t_arity=t_arity, time_stamp=time_stamp)
        self.size_in = in_size
        self.rule_number = rule_number
        self.alpha = default_alpha
        self.number_aux_pred = number_aux_pred
        self.ini_layer = AndLayer(number_features=self.size_in, number_aux_pred=self.number_aux_pred)
        self.middle_layer = AndLayer(number_features=self.number_aux_pred, number_aux_pred=self.rule_number)
        self.last_layer = OrLayer()
        self.device = device
        self.output_rule_file = output_rule_file
        self.t_relation = t_relation 
        self.task = task
    
    def forward(self, x):
        '''
        x indicate all feature values in the data
        '''
        activated_aux_predicates = self.ini_layer(x)
        activated_head_predicates = self.middle_layer(activated_aux_predicates)
        rule_satisfy = self.last_layer(activated_head_predicates)
        return rule_satisfy # let the rule satisfy be 1 always 
    
    def interpret_weights_computation(self):
        '''
        return the interpretable matrix based on the weights of the model 
        '''
        full_features_weights = self.ini_layer.interpretable_rul_weights
        aux_predicates_weights = self.middle_layer.interpretable_rul_weights
        each_rule_activation = torch.mm(aux_predicates_weights.t(), full_features_weights.t())
        final_interpret = each_rule_activation # number_rules X number_features
        self.current_final_interpret = final_interpret
        return final_interpret
        
    
    def get_invented_predicate_v2(self):
        middle_weights = self.middle_layer.interpretable_rul_weights.cpu().detach().numpy()
        middle_weights = np.transpose(middle_weights)
        front_weights = self.ini_layer.interpretable_rul_weights.cpu().detach().numpy()
        front_weights = np.transpose(front_weights)
        
        interpret_invented_weighs = []
        for rule_index, predicated_item in enumerate(middle_weights):
            index = 0
            weighted_features = []
            for weights in predicated_item:
                weighted_features.append(weights * front_weights[index,:])
                index += 1
            weighted_features = np.array(weighted_features)
            saved_atoms = []
            for threshold in range(1,100):
                threshold = threshold/100
                activated_index = np.where(weighted_features > threshold)
                
            
                pairs_rule_atoms = []
                invented_predicate = collections.defaultdict(list) # store the rule index and 
                index_rules = list(activated_index[0])
                values_items = list(activated_index[1])

                for index, value in enumerate(index_rules):
                    corresponding_atoms = self.atoms[values_items[index]]
                    invented_predicate[value].append(corresponding_atoms)
                    if corresponding_atoms not in pairs_rule_atoms:
                        pairs_rule_atoms.append(corresponding_atoms)
                if pairs_rule_atoms == []:
                    continue
                predicate_list = []
                for item in pairs_rule_atoms:    
                    predicate_list.append(Predicate(Feature(item), Relation.EQ, 1))
                conjunct = Conjunction(predicate_list)
                rules_obj = DnfRuleSet([conjunct], 1)
                print(rules_obj)
                
                # check the precision 
                me_obj = self.check_metric(test_data=None, rule_set=rules_obj, remove_low_precision=None,print_result=False)
                precision = me_obj['metrics']['rule_set_precision']
                if precision != 1:
                    continue
                else:
                    if str(conjunct) not in saved_atoms:
                        saved_atoms.append(str(conjunct))
                        interpret_invented_weighs.append({'rule':invented_predicate, 'thres':threshold, 'rule_index':rule_index})
                    else:
                        continue
        interpret_invented_weighs = self.check_correctness_invented_predicate(interpret_invented_weighs)
        # print(interpret_invented_weighs)
        with open(f'{self.output_rule_file}', 'a+') as f:
            for item in interpret_invented_weighs:
                print(f'success invented predicate:**{item}**',file=f)
        return interpret_invented_weighs
                
    def check_correctness_invented_predicate(self, interpret_invented_weighs, test_data:pd.DataFrame=None):
        '''
        Assume the invented predicate is inv(X,Y),
        choose the variable which also appear in another set of atoms
        add constraints that only two atoms can be activated at the same time
        '''
        corrected_invented_predicates = []
        for rule_index in interpret_invented_weighs:
            and_rules = rule_index['rule']
            unique_body = set([])
            for invented_index, invented_body in and_rules.items():
                invented_body = frozenset(invented_body)
                unique_body.add(invented_body)
                
            if len(unique_body) == 1:
                continue
            else:
                KG_checker = CheckMetrics(t_relation=self.t_relation, task_name=self.task, classifier=None, logic_path=self.output_rule_file, ruleset=unique_body,t_arity=self.t_arity)
                corrected_flag = KG_checker.check_logic_program_with_IP()
                if corrected_flag == 1:
                    corrected_invented_predicates.append(rule_index)
        return corrected_invented_predicates
        
                
    
    def interpret_invented_predicate(self, best_threshold,removed_or_list=None):
        '''
        call this function just after the interpret function to get the invented predicate
        '''
        current_final_interpret = self.interpret_weights_computation()
        interpretable_weights_cpu = current_final_interpret.cpu().detach().numpy()
        if removed_or_list != None:
            for item in removed_or_list:
                interpretable_weights_cpu[item,:] = -1
        activated_index = np.where(interpretable_weights_cpu > best_threshold)
        pairs_rule_atoms = collections.defaultdict(list) # store the rule index and highlighted atoms [[item](rule_index)...[]]
        index_rules = list(activated_index[0])
        values_items = list(activated_index[1])

        for index, value in enumerate(index_rules):
            corresponding_atoms_index = values_items[index]
            pairs_rule_atoms[value].append(corresponding_atoms_index)
        # for the end to front to extract rules with predicated atoms
        # can missing the body atoms extraction (front atoms) if the predicated atoms (end atoms) are not big enough
        
        middle_weights = self.middle_layer.interpretable_rul_weights.cpu().detach().numpy()
        middle_weights = np.transpose(middle_weights)
        front_weights = self.ini_layer.interpretable_rul_weights.cpu().detach().numpy()
        front_weights = np.transpose(front_weights)
        all_possible = []
        all_possible_string = []
        for rule_index, predicated_item in enumerate(middle_weights):
            if rule_index in removed_or_list:
                continue
            all_DNF_rule_set = []
            all_rules_str_no_head = []
            for middle_threshold in range(1,100):
                single_invented_rules = set([])
                invented_predicate_list = []
                middle_threshold = middle_threshold/100
                activated_middle_atoms = np.where(predicated_item > middle_threshold)[0]
                if len(activated_middle_atoms) < 2:
                    continue
                for invented_index in activated_middle_atoms:
                    # the invented predicate only got two states for activated or not
                    if predicated_item[invented_index] * 100 < 1:
                        continue
                    single_invented_rules_list = []
                    
                    for frond_threshold in range(1,100):
                        predicated_list = []
                        frond_threshold = frond_threshold/100
                        activated_atoms = np.where(front_weights[invented_index,:] > frond_threshold)[0]
                        for body_a in activated_atoms:
                            pred_atom = self.atoms[body_a]
                            predicated_list.append(Predicate(Feature(pred_atom), Relation.EQ, 1))
                        if predicated_list == []:

                            continue
                        conjunct = Conjunction(predicated_list)
                        if str(conjunct) not in single_invented_rules:
                            single_invented_rules.add(str(conjunct))
                            single_invented_rules_list.append(conjunct)
                    if single_invented_rules_list == []:
                        continue
                    predicated_rule = DnfRuleSet(single_invented_rules_list, f'inv_atom_{invented_index}')
                    invented_no_head_rule = DnfRuleSet(single_invented_rules_list, f'inv_atom') # no head for checking duplicate
                    invented_predicate = Predicate(Feature(f'inv_atom_{invented_index}'), Relation.EQ, 1)
                    if str(invented_no_head_rule) not in all_rules_str_no_head:
                        all_rules_str_no_head.append(str(invented_no_head_rule))
                        all_DNF_rule_set.append(predicated_rule)
                        invented_predicate_list.append(invented_predicate)
                
                if invented_predicate_list != []:
                    con_invented = Conjunction(invented_predicate_list)
                    invented_rule = DnfRuleSet([con_invented], 1)
                    all_DNF_rule_set.append(invented_rule)

                string_all_set = ''
                for item in all_DNF_rule_set:
                    string_all_set += str(item) + '\n'
                if string_all_set not in all_possible_string:
                    if '(true)' in string_all_set:
                        break
                    all_possible_string.append(string_all_set)
                    all_possible.append(all_DNF_rule_set)
        with open(f'{self.output_rule_file}_ip.md', 'a+') as f:
            for item in all_possible:
                for i in item:
                    print(i,file=f)
                print('-----------------',file=f)
        self.all_possible = all_possible
        return all_possible
    
    def check_invented_precision_recall(self):
        new = []
        for item in self.all_possible:
            print(item)
            metrixs = CheckMetrics(t_relation=self.t_relation, task_name=self.task, classifier=None, logic_path=None, ruleset=item,t_arity=self.t_arity)
            metrixs.check_logic_program_with_IP()
            if metrixs == 1:
                keep = True
            else:
                keep = False
            if keep == True:
                new.append(item)
        with open(f'{self.output_rule_file}_ipm.md', 'a+') as f:
            for item in new:
                for i in item:
                    print(i,file=f)
            print('-----------------',file=f)
        return 0 
    
        
    def check_best_rules_with_threshold(self):
        '''
        Obtain the threshold when the rule is best 
        '''
        current_final_interpret = self.interpret_weights_computation()
        for threshold_step in range(1,100):
            rule_set = None
            string_predicates = set([])
            all_conjunctions = []
            threshold = threshold_step/100
            interpretable_weights_cpu = current_final_interpret.cpu().detach().numpy()
            activated_index = np.where(interpretable_weights_cpu > threshold)
            pairs_rule_atoms = collections.defaultdict(list) # store the rule index and highlighted atoms [[item](rule_index)...[]]
            index_rules = list(activated_index[0])
            values_items = list(activated_index[1])

            for index, value in enumerate(index_rules):
                corresponding_atoms = self.atoms[values_items[index]]
                pairs_rule_atoms[value].append(corresponding_atoms)

            for key, item_list in pairs_rule_atoms.items():
                predicate_list = []
                for item in item_list:
                    predicate_list.append(Predicate(Feature(item), Relation.EQ, 1))
                conjunct = Conjunction(predicate_list)
                if str(conjunct) not in string_predicates:
                    string_predicates.add(str(conjunct))
                    all_conjunctions.append(conjunct)
            rule_set = DnfRuleSet(all_conjunctions, 1)
            print(rule_set)
            # check metrics 
            return_obj = self.check_metric(test_data=None, rule_set=rule_set, remove_low_precision=0.8)
            metrics, rule_set = return_obj['metrics'], return_obj['rule_set']
            removed_lit = return_obj['removed_list']
            if metrics['rule_set_precision'] == 1:
                return threshold, removed_lit
            
        return 0 , None

class DeepRuleLayer(BaseRule):
    def __init__(self, in_size, rule_number, default_alpha = 10, device='cuda:0',output_rule_file = './a_rule.md', t_relation = '', task = '',t_arity=2,number_aux_pred=5, time_stamp=''):
        '''
        currently tis  single layer
        '''
        super().__init__(in_size, rule_number, default_alpha, device, output_rule_file,t_arity=t_arity, time_stamp=time_stamp)
        self.size_in = in_size
        self.rule_number = rule_number
        self.alpha = default_alpha
        self.number_aux_pred = number_aux_pred
        self.ini_layer = AndLayer(number_features=self.size_in, number_aux_pred=2, default_alpha=1)
        # self.droplayer = nn.Dropout(p=0.5)
        # self.middle_layer = AndLayer(number_features=2, number_aux_pred=2, default_alpha=5)
        # self.third_layer = AndLayer(number_features=5, number_aux_pred=2)
        self.and_series  = nn.Sequential(
            self.ini_layer,
        )
        self.and_layers = nn.ModuleList([self.ini_layer])
        self.last_layer = OrLayer()
        self.device = device
        self.output_rule_file = output_rule_file
        self.t_relation = t_relation 
        self.task = task
    
    def forward(self, x):
        '''
        x indicate all feature values in the data
        '''
        x = self.and_series(x)
        rule_satisfy = self.last_layer(x)
        return rule_satisfy # let the rule satisfy be 1 always 
    
    def interpret_weights_computation(self):
        '''
        return the interpretable matrix based on the weights of the model 
        '''
        final_interpret = self.and_layers[0].interpretable_rul_weights
        for layers in self.and_layers[1:]:
            final_interpret = torch.mm(final_interpret, layers.interpretable_rul_weights)
        final_interpret = final_interpret.t()
        self.current_final_interpret = final_interpret
        return final_interpret
    
    
class DeepResidualLogic(BaseRule):
    '''
    The learning predicate invention 
    Similar with recurrent neural networks 
    '''
    def __init__(self, in_size, rule_number, default_alpha = 10, device='cuda:0',output_rule_file = './a_rule.md', t_relation = '', task = '',t_arity=2, residual_layer=1, number_aux_pred = 3, time_stamp=''):
        '''
        residual_layer: the number of residual layer except the last and layer
        '''
        super().__init__(in_size, rule_number, default_alpha, device, output_rule_file, t_relation, task, t_arity=t_arity, number_aux_pred=number_aux_pred, time_stamp=time_stamp)
        self.middle_layers = []
        self.ini_layer = AndLayer(number_features=self.size_in, number_aux_pred=self.number_aux_pred).to(device)
        for i in range(residual_layer):
            self.middle_layers.append(AndLayer(number_features=self.size_in+self.number_aux_pred, number_aux_pred=self.number_aux_pred).to(device))
        self.middle_layers.append(AndLayer(number_features=self.size_in+self.number_aux_pred, number_aux_pred=self.rule_number).to(device))
        self.last_layer = OrLayer().to(device)
        
    def forward(self,x):
        activated_aux_predicates = self.ini_layer(x)
        for layer in self.middle_layers:
            composed_state = torch.cat((x, activated_aux_predicates), 1)
            activated_aux_predicates = layer(composed_state)
        rule_satisfy = self.last_layer(activated_aux_predicates)
        return rule_satisfy
    
    def interpret_weights_computation(self):
        '''
        return the interpretable matrix based on the weights of the model 
        begin from the first layer 
        '''
        full_features_weights = self.ini_layer.interpretable_rul_weights
        for middle_layer in self.middle_layers:
            aux_middle_layer = middle_layer.interpretable_rul_weights[self.size_in:,:]
            residual_weights = middle_layer.interpretable_rul_weights[0:self.size_in,:]
            full_features_weights =  torch.mm(full_features_weights,aux_middle_layer) + residual_weights
        self.current_final_interpret = full_features_weights
        return full_features_weights
        
        
    
    def get_invented_predicate_v2(self):
        pass
    # interpret the invented predicate and 
    # try plot one 

class DNN_normal(CoreLayer):
    def __init__(self, input):
        super().__init__(in_size=input, rule_number=1)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    def interpret_weights_computation(self):
        softmax_layers = []
        for layer in self.linear_relu_stack:
            if type(layer) == nn.Linear:
                weights = layer.weight
                softmax_layers.append(torch.nn.functional.softmax(weights, dim=1))
        self.interpretable_rul_weights = softmax_layers[0].t()
        for layer in softmax_layers[1:]:
            self.interpretable_rul_weights = torch.mm(self.interpretable_rul_weights, layer.t())
        self.interpretable_rul_weights = self.interpretable_rul_weights.t() 
        return self.interpretable_rul_weights


if __name__ == '__main__':
    # test the model 
    data = pd.read_csv('gammaILP/logic_back/demo_data.csv')
    all_atoms = list(data.columns)
    all_atoms.remove('label')
    model = DeepRuleLayer(in_size=5, rule_number=2, default_alpha=10, device='cuda:0', output_rule_file='gammaILP/logic_back//a_rule.md', t_relation='', task='', t_arity=2, number_aux_pred=5, time_stamp='')
    values = data.values
    train_data = DataLoader(values, batch_size=10, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    model.train()
    model.to('cpu')
    for epoch in range(100):
        for batch in train_data:
            x = batch[:, :-1].float().to('cpu')
            y = batch[:, -1].float().to('cpu')
            optimizer.zero_grad()
            outputs = model(x)
            outputs = outputs.view(-1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    rules = model.interpret(all_atoms, 100)
    
# todo 
# 1. PRetrian the body loss 
# 2. Trian all togetger 




class CheckMetrics():
    '''
    When the data base is in relational KG or logic rule is in first-order logic, check the acc from this code 
    '''
    def __init__(self,t_relation, task_name, logic_path,t_arity=2, test_mode=False, ruleset: DnfRuleSet = None, data_path = 'rule_learning_original/code/DFORL/\{tasl_name\}/data/',all_relation=None,relational_image_test=False):
        '''
        - Need to make classifier through a rule_set data in the format of DnfRuleSet (from aix360_k.aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet)
        - Initial code:
        classifier = trxf_classifier.RuleSetClassifier([rule_set],rule_selection_method=trxf_classifier.RuleSelectionMethod.FIRST_HIT,confidence_metric=trxf_classifier.ConfidenceMetric.LAPLACE,weight_metric=trxf_classifier.WeightMetric.CONFIDENCE,default_label=0)
        @param
        data_path: the facts file end with .nl with all testing facts 
        '''
        self.t_relation = t_relation 
        self.task_name = task_name
        self.test_mode = test_mode
        self.data_path = data_path
        self.logic_path = logic_path
        self.all_relation = all_relation
        self.hit_flag = False
        self.t_arity=t_arity
        self.hit_test_predicate = []
        self.rule_set = ruleset
        self.classifier = trxf_classifier.RuleSetClassifier([self.rule_set],rule_selection_method=trxf_classifier.RuleSelectionMethod.FIRST_HIT,confidence_metric=trxf_classifier.ConfidenceMetric.LAPLACE,weight_metric=trxf_classifier.WeightMetric.CONFIDENCE,default_label=0)
        if relational_image_test == True:
            self.facts_path = self.data_path+self.t_relation+'.nltest'
        else:
            self.facts_path = self.data_path+self.t_relation+'.nl'
        self.all_variables = ['X','Y','Z','W','M','N','T','U','V','A','B','C','D','E','F','G','H','I','J','K','L','O','P','Q','R','S']
        self.all_variables_str = 'X,Y,Z,W,M,N,T,U,V,A,B,C,D,E,F,G,H,I,J,K,L,O,P,Q,R,S,'
        if self.t_arity == 2:
            self.first_variable = 'X'
            self.second_variable = 'Y'
        elif self.t_arity == 1:
            self.first_variable = 'X'
            self.second_variable = 'X'
        else:
            print('The arity of the target relation is not supported')
        
    def  build_target_predicate(self, left_bracket='(', right_bracket=')', split=',',end_symbol='.'):
        '''
        build target predicate and return a dictionary consisting the predicate 
        param:
        left_bracket: the left bracket symbol in knowledge facts set
        right_bracket: the right bracket symbol in knowledge facts set
        split: the split symbol in knowledge facts set
        end_symbol: the end symbol in knowledge facts set
        '''
        all_target_predicate_dic = {}
        with open(self.facts_path, 'r') as f:
            single_line = f.readline()
            while single_line:
                # skip the negative examples in the datasets, we won't add the negative examples in test datasets
                if '-' in single_line:
                    single_line = f.readline()
                    continue
                if self.test_mode == True:
                    # if the test mode is True, then check the accuracy based on the test datasets 
                    if 'TEST' in single_line:
                        head_relation = single_line[:single_line.index(left_bracket)]
                        if head_relation == self.t_relation:
                            all_target_predicate_dic[single_line[:single_line.index(right_bracket)+1]] = 0
                else:
                    # if the test mode is Off, then check auucracy expecting the test datasets
                    if 'TEST' in single_line:
                        single_line = f.readline()
                        continue
                    head_relation = single_line[:single_line.index(left_bracket)]
                    if head_relation == self.t_relation:
                        all_target_predicate_dic[single_line[:single_line.index(right_bracket)+1]] = 0
                single_line = f.readline()
            f.close()
        self.all_target_predicate_dic = all_target_predicate_dic
        return all_target_predicate_dic
    
    def check_correctness_of_logic_program(self, test_mode = False, left_bracket='(', right_bracket=')', split=',',end_symbol='.', split_atom='_',left_atom='[', right_atom=']', minimal_precision = 0):
        '''
        Input the symbolic logic program, return the correctness (precision and recall) of each rule in the logic prgoram. 
        This process do not delete any rules from the existing logic program. Only check the recall and precision based on the existing minimal precision constraints.
        param:
        left_bracket: the left bracket symbol in knowledge facts set 
        right_bracket: the right bracket symbol in knowledge facts set
        split: the split symbol in knowledge facts set
        end_symbol: the end symbol in knowledge facts set
        split_atom: the split symbol in logic rules 
        '''
        # expresison_list, variable_index, head_relation = build_datalog_base_and_expression(data_path, t_relation, logic_program_name, task_name, result_path)
        if type(self.all_relation) != list:
            relation_path = self.data_path + 'all_relation_dic.dt'
            with open(relation_path,'rb') as f:
                all_relation = pickle.load(f)
                f.close()
        else:
            all_relation = self.all_relation
        # if classifer is None, then there is no rule
        try:
            rule_list = self.classifier.rules
        except:
            rule_list = []
        head_relation = self.t_relation
        precision_rule = [] 
        body = []
        for rule in rule_list:
            if self.hit_flag == True:
                precision_rule.append(rule.precision)
            body.append(str(rule.conjunction))
                

        # ! build the variable into Datalog
        term_log = ''
        # term_log += 'X,Y,Z,W,M,N,T,'
        term_log += self.all_variables_str
        for i in all_relation:
            if i == '':
                continue
            term_log += i + ','
        # remove the comma at the end of the string in term_log
        index_end = len(term_log) - 1
        while index_end >= 0:
            if term_log[index_end] == ',':
                index_end -= 1
            else:
                break
        # remove the last comma in the term_log
        term_log = term_log[:index_end+1]
        # term_log = term_log[0:-1]
        pyDatalog.create_terms(term_log)
        
        with open(self.facts_path, 'r') as f:
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
                    continue
                #prepare objects
                single_tri = single_tri[:single_tri.index(end_symbol)]
                relation_name = single_tri[:single_tri.index(left_bracket)]
                first_entity = single_tri[single_tri.index(left_bracket) + 1 : single_tri.index(split)]
                second_entity = single_tri[single_tri.index(split)+1 : single_tri.index(right_bracket)]
                #add to database
                + locals()[relation_name](first_entity,second_entity)
                
                single_tri = f.readline()
            f.close()
        
        # Check each generated rules 
        expresison_list = [None] * len(body)
        variable_index = []
        rule_index_append = 0
        for rules in body:
            # Find the order of variables x and y and z in the rules
            va_f = 0
            o_variable_index = []
            for var in self.all_variables:
            # for var in ['X','Y','Z','W','M','N','T']:
                if var in rules:
                    index = rules.index(var)
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
            item_list = rules.replace(' ','').split('^')
            expression = []
            variable_number = 0
            for position in o_variable_index:
                if position[1] != 1e6:
                    variable_number += 1
                if position[0] == self.first_variable:
                    self.first_variable_index = position[1]
                if position[0] == self.second_variable:
                    self.second_variable_index = position[1]
            if self.first_variable_index == 1e6 or self.second_variable_index == 1e6:
                expresison_list[rule_index_append] = []
                rule_index_append += 1
                continue
            #! safety check for the variable number
            if variable_number > 5:
                expresison_list[rule_index_append] = []
                rule_index_append += 1
                continue

            for item in item_list:
                item = item[1:-1]
                negation_flag = False
                if item == '':
                    continue
                else:
                    name = item[:item.index(left_atom)]
                    if name == '':
                        continue
                    # if the neagation operator in the rule 
                    if '~' in name:
                        negation_flag = True
                        name = name[1:]
                    first_variable = item[item.index(left_atom)+1: item.index(split_atom)].upper()
                    second_variable = item[item.index(split_atom)+1: item.index(right_atom)].upper()
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
            # if flag != 0:
            expresison_list[rule_index_append] = expression
            rule_index_append += 1
        # ! each expression corresponds a rule
        # read the target predicate file 
        if self.hit_flag == True:
            target_dic = self.hit_test_predicate
        else:
            target_dic = self.build_target_predicate(left_bracket=left_bracket, right_bracket=right_bracket, split=split, end_symbol=end_symbol)
        
        # The following check process follows the T_P operator of the logic program 
        correct_f = []
        search_index = 0 
        invalid_rules = []
        expresison_list = list(expresison_list)
        for res in expresison_list: # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
            num_validate = 0
            correct = 0
            last_tar_dic = target_dic.copy()
            for re in res:
                x_index = variable_index[search_index][self.first_variable]
                y_index = variable_index[search_index][self.second_variable]
                if x_index >= len(re) or y_index >= len(re):
                    break
                num_validate += 1
                if self.t_arity == 2:
                    first_res = re[x_index]
                    # if first_res == 'iraq':
                    #     b = 90
                    second_res = re[y_index]
                elif self.t_arity == 1:
                    first_res = re[x_index]
                    second_res = re[x_index]
                final = len(locals()[head_relation](first_res,second_res)) # The ground predicate 
                # ! test mode open iff when check accuracy basedon the .nl file
                if test_mode == True:
                    predicate = head_relation + left_bracket + first_res+ split +second_res+ right_bracket
                    if predicate in target_dic:
                        if self.hit_flag == True:
                            current_precision_value = target_dic[predicate]
                            # iff the new precision are larger than the current one, updata the value 
                            if precision_rule[search_index] >= current_precision_value: 
                                target_dic[predicate] = precision_rule[search_index]
                        else:
                            target_dic[predicate] = 1
                        correct += 1    
                # ! when the test mode is false and the target ground predicate in the database
                elif final == 1:  
                    predicate = head_relation + left_bracket + first_res+ split +second_res+ right_bracket
                    if predicate in target_dic:
                        target_dic[predicate] = 1
                    correct += 1
            if num_validate == 0:
                num_validate = -1
            # precision_rule = correct/num_validate
            if correct/num_validate < minimal_precision:
                target_dic = last_tar_dic.copy()
                invalid_rules.append(search_index)
            correct_f.append((correct/num_validate ,correct, num_validate))
            search_index += 1
            last_tar_dic = target_dic.copy()
        # print(correct_f)
        # write the state of target predicate into the disk 
        # When executing single task, writing each test predicate  logic in the disk 
        # if self.hit_flag == False:
        #     write_target_predicate(task_name, target_dic, t_relation)
        
        # keep the valid rule and precisions 
        
        valid_rule_index = [i for i in range(len(correct_f)) if i not in invalid_rules]
        valid_correct_f = []
        for i in valid_rule_index:
            valid_correct_f.append(correct_f[i])
            
        false_instance = 0
        for key in target_dic:
            if target_dic[key] == 0:
                false_instance += 1
        recall_KG = (len(target_dic) - false_instance)/len(target_dic)
        # if correct_f != [] and recall_KG != 0:
        #     with open(self.logic_path, 'w') as f:
        #         print('**precision** from KG:', valid_correct_f, file= f)
        #         print(f'**recall** from KG:**{recall_KG}**', file=f)
        #         f.close()
                

        
        if self.hit_flag == True:
            correct_f = target_dic
        return_obj  = {'precision':correct_f, 'recall':recall_KG}
        valid_return_obj = {'precision':valid_correct_f, 'recall':recall_KG}

        with open(self.logic_path, 'w+') as f:
            ini_index = 0
            for rule in rule_list:
                if return_obj['precision'][ini_index][0] >= 0 and return_obj['precision'][ini_index][0] >= minimal_precision:
                    print('', str(rule.conjunction), file=f, end=' ')
                    print(correct_f[ini_index], file=f)
                ini_index += 1
            # print return obj
            ini_index = 0
            print('Precision: ',file=f,end = '')
            for rule_pre in valid_return_obj['precision']:
                if rule_pre[0] > 0:
                    print(rule_pre, file=f, end=' ')
            print(file=f)
            print('Recall: ', valid_return_obj['recall'],file=f)
            f.close()
        return valid_return_obj

    def check_logic_program_with_IP(self, test_mode = False):
        '''
        check the inventied logic program with the IP operator
        '''
        # expresison_list, variable_index, head_relation = build_datalog_base_and_expression(data_path, t_relation, logic_program_name, task_name, result_path)
        
        relation_path = self.data_path + 'all_relation_dic.dt'
        with open(relation_path,'rb') as f:
            all_relation = pickle.load(f)
            f.close()
        # when check from predicated invention need to do that 
        rule_list = self.rule_set
        head_relation = self.t_relation
        body = ''
        for rule in rule_list:
            all_conjuntions = rule
            for conj in all_conjuntions:
                body += (str(conj))+'^'
        body = [body[:-1]]
        # ! build the variable into Datalog
        term_log = ''
        term_log += self.all_variables_str
        # add existing predicates
        for i in all_relation:
            term_log += i + ','

        
        term_log = term_log[0:-1]
        pyDatalog.create_terms(term_log)
        
        with open(self.facts_path, 'r') as f:
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
                    continue
                #prepare objects
                single_tri = single_tri[:single_tri.index('.')]
                relation_name = single_tri[:single_tri.index('(')]
                first_entity = single_tri[single_tri.index('(') + 1 : single_tri.index(',')]
                second_entity = single_tri[single_tri.index(',')+1 : single_tri.index(')')]
                #add to database
                + locals()[relation_name](first_entity,second_entity)
                
                single_tri = f.readline()
            f.close()
        
        # Check each generated rules 
        expresison_list = []
        variable_index = []
        for rules in body:
            # Find the order of variables x and y and z in the rules
            va_f = 0
            o_variable_index = []
            for var in self.all_variables:
                if var in rules:
                    index = rules.index(var)
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
            item_list = rules.replace(' ','').split('^')
            for item in item_list:
                negation_flag = False
                if item == '':
                    continue
                else:
                    name = item[:item.index('[')]
                    # if the neagation operator in the rule 
                    if '~' in name:
                        negation_flag = True
                        name = name[1:]
                    first_variable = item[item.index('[')+1: item.index(',')].upper()
                    second_variable = item[item.index(',')+1: item.index(']')].upper()
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
        if self.hit_flag == True:
            target_dic = self.hit_test_predicate
        else:
            target_dic = self.build_target_predicate()
        
        # derived atoms from then invented predicate 
        derived_atoms = []
        for res in expresison_list:
            for re in res:
                x_index = variable_index[0]['X']
                y_index = variable_index[0]['Y']
                if x_index >= len(re) or y_index >= len(re):
                    break
                if self.t_arity == 2:
                    first_res = re[x_index]
                    second_res = re[y_index]
                elif self.t_arity == 1:
                    first_res = re[x_index]
                    second_res = re[x_index]
                derived_atoms.append(self.t_relation+f'({first_res},{second_res})')
        if set(derived_atoms).issubset(set(target_dic.keys())):
            print('invented predicate success')
            return 1
        else:
            print('invented predicate not success')
            return 0
        # The following check process follows the T_P operator of the logic program 
        # correct_f = []
        # search_index = 0 
        # for res in expresison_list: # expression_list: [[[g(x),g(y),g(z)],[g(x),g(y),g(z)],[g(x),g(y),g(z)]]...]
        #     num_validate = 0
        #     correct = 0
        #     for re in res:
        #         x_index = variable_index[search_index]['X']
        #         y_index = variable_index[search_index]['Y']
        #         if x_index >= len(re) or y_index >= len(re):
        #             break
        #         num_validate += 1
        #         if self.t_arity == 2:
        #             first_res = re[x_index]
        #             # if first_res == 'iraq':
        #             #     b = 90
        #             second_res = re[y_index]
        #         elif self.t_arity == 1:
        #             first_res = re[x_index]
        #             second_res = re[x_index]
        #         final = len(locals()[head_relation](first_res,second_res)) # The ground predicate 
        #         # ! test mode open iff when check accuracy basedon the .nl file
        #         if test_mode == True:
        #             predicate = head_relation + '(' + first_res+',' +second_res+ ')'
        #             if predicate in target_dic:
        #                 target_dic[predicate] = 1
        #                 correct += 1    
        #         # ! when the test mode is false and the target ground predicate in the database
        #         elif final == 1:  
        #             predicate = head_relation + '(' + first_res+',' +second_res+ ')'
        #             if predicate in target_dic:
        #                 target_dic[predicate] = 1
        #             correct += 1
        #     if num_validate == 0:
        #         num_validate = -1
        #     correct_f.append((correct/num_validate ,correct, num_validate))
        #     search_index += 1
        # print(correct_f)
        # # write the state of target predicate into the disk 
        # # When executing single task, writing each test predicate  logic in the disk 
        # # if self.hit_flag == False:
        # #     write_target_predicate(task_name, target_dic, t_relation)
        # false = 0
        # for key in target_dic:
        #     if target_dic[key] == 0:
        #         false += 1
        # recall_KG = (len(target_dic) - false)/len(target_dic)
        # with open(self.logic_path, 'a') as f:
        #     print('**precision** from KG:', correct_f, file= f)
        #     print(f'**recall** from KG:**{recall_KG}**', file=f)
        #     f.close()
            
        # # if hit == True:
        # #     with open(data_path +  t_relation + '/relation_entities.dt', 'rb') as f:
        # #         relation_entity = pickle.load(f)
        # #         f.close()
        # #     res = calculate_Hits(target_tuple,expresison_list, correct_f, variable_index)
        # #     print("Hits result:", res)
        
        # if self.hit_flag == True:
        #     correct_f = target_dic
        # return correct_f
