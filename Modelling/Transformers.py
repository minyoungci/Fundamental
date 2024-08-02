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
