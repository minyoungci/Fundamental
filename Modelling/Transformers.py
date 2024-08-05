import torch
import torch.nn as nn
import math

# PositionalEncoding 클래스 정의
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int ): # (1) 벡터의 차원을 d_model로 받아옴, vocab_size는 단어의 개수
        super().__init__()  
        self.d_model = d_model  
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # 임베딩 레이어 생성 

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) # (2) 매번 동일한 벡터에 숫자를 매핑하는 역할 ---------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float)-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # seq_len x d_model 크기의 행렬 생성
        pe = torch.zeros(seq_len, d_model)
        
        # seq_len 크기의 벡터 생성  (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # sin을 짝수 인덱스에 적용 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)    

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)    

        self.register_buffer('pe', pe) # pe를 모델의 파라미터로 저장하지 않고, 버퍼로 저장

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (2) 임베딩 벡터에 positional encoding을 더함
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 **-6) -> None :
        super().__init__()
        self.eps = eps # 수치 안정을 위한 매우 작은 값
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added 

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True) 
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

