from lm_as_kb import main, argparser, data, trainer
import torch  
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Propositionztion(nn.Module):
    '''
    Not a trainable model
    '''
    def __init__(self):
        self.arg_lm()
        self.data = self.data_make()

        
    def arg_lm(self):
        
        conf = argparser.get_conf()
        conf.command = 'test'
        conf.top_n = 7128 #
        conf.n_facts = 1 #
        conf.max_test_inst = 1196 #
        conf.max_eval_inst = 1
        conf.entity_repr = 'symbol'
        conf.architecture = 'lstm'
        conf.n_layers = 2
        conf.n_hidden = 1024
        conf.lr = 0.001
        conf.batch_size = 128
        conf.km_emb_random_init = True 
        conf.model_file = Path('gammaILP/lm_as_kb/out/67/checkpoint_acc=0.7450.pt')
        conf.early_stopping = 40
        conf.test_data_path = f'cache/icews14/relation_statements.en.top{conf.top_n}.jl'
        conf.batch_size = 128
        self.conf = conf
        
        
    def forward(self):
        inter = self.make_interpretation()
        # train_data = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        # self.make_interpretation(train_data)
        # self.differentiable_rule_learner()
        
    def make_interpretation(self):
        # make interpretation
        interpretation = self.lm_model.test()
        
        return interpretation
        
        
    def data_make(self):
        # data loader 
        '''
        input tuple of elements in a batch 
        '''
        self.lm_model = trainer.Trainer(self.conf)
        
    

if __name__ == '__main__':
    model = Propositionztion()
    model.forward()