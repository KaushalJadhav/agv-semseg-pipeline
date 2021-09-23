import torch.nn as nn 

class CrossEntropyLoss(nn.Module):
    def __init__(self, config=None):
        super(CrossEntropyLoss,self)._init_()
        if config==None:
            self.loss=nn.CrossEntropyLoss()
        else:
            self.loss=nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    
    def forward(self,input,tagrets):
        return self.loss(inputs,targets)
