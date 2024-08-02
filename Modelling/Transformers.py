import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int ): # (1) 벡터의 차원을 d_model로 받아옴, vocab_size는 단어의 개수
        super().__init__()  
        self.d_model = d_model  
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # 임베딩 레이어 생성

    def forward(self,x):
        return self.embedding(x)
