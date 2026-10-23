"""Historical B2-C1 optimizer recipe; constructing it performs no step."""
import math,torch

def lr_at(index):
 assert 0<=index<2880
 return 1e-5+.5*(2e-4-1e-5)*(1+math.cos(math.pi*index/2879))

def optimizer(model):
 return torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
