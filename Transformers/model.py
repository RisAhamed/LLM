import torch
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):

    def __init__(self,d_model: int,vacob_size : int):
        super().__init__()
        self.d_model = d_model
        self.vacob_size = vacob_size
        self.embedding = nn.Embedding(vacob_size,d_model)

    def forward(self,x):
        return self.emmbedding(x) * math.sqrt(self.d_model)


class positionalEmbeddings(nn.Module):
    def __init__(self,d_model: int ,seq_len: int,droupout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # matrix of shape(seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)

        # Vector of shape(seq_len,1)
        position  = torch.arange(0,seq_len,dtype =torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:,0::2] = torch.sin(position* div_model)
        pe[:,1::2] = torch.cos(position* div_model)
        pe = pe.unsqeeze(0)
        self.registerbuffer('pe',pe)


    def forward(self,):
        x = x+ (self.pe[: ,: xx.shape[1],:]).requires_grad(False)
        return self.droupout(x)


class Layernorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        # self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        x = x.mean(dim =-1,keepdim =True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma  * (x-mean) /(self.eps + std) + self.beta

class FeedForwardBlock(nn.Moddule):

    def __init__(self, d_model: int, dff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1  = nn.Linear(d_model, dff, )
        # self.dff = dff
        self.dropout = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(d_model,dff)
        # self.fc2 = nn.Linear(dff,d_model)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self, x):
        return self.linear_2(self.droupout(torch.relu(self.linnear_1(x))))

class MultiheadAttention(nn.Module):

    def __init__(self,d_model: int,h: int,dropout: float)-> None:
        super().__int__()
        self.d_model = d_model
        self.h = h
        assert d_model % h ==0

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)   
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)
        self.droupout = nn.Dropout(d_model,d_model)

    def attention(query,key,value,mask):
        pass
    def forward(self,q,k,v,mask):
        query   = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1,2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1,2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1,2)
        