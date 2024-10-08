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

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int , d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2    

    def forward(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, h: int, dropout: float) -> None: 
        super().__init__()
        self.d_model = d_model 
        self.h = h 
        assert d_model % h ==0 , "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv 

        self.w_o = nn.Linear(d_model, d_model) # wo
        self.dropout = nn.Dropout(dropout)  

    @staticmethod # 클래스의 인스턴스 생성 없이 바로 접근 가능
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]


        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len) 
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # (Batch, h, seq_len, d_k) x (Batch, h, d_k, seq_len) = (Batch, h, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9) # mask가 0인 부분에 -1e9를 넣어줌
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores 


    def forward(self, q, k, v, mask = None):
        query = self.w_q(q) #(Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k)   #(Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v) #(Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        # (Batch, seq_len, d_model) -> ( Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)    
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = query.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) 
        
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)   
        return self.w_o(x) 


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None :
        super().__init__()
        self.dropout = nn.Dropout(dropout)  
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block    
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask): 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x 

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x) 

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.src_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])   

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) 
        x = self.residual_connections[1](x, lambda x: self.src_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size :int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch,seq_len,d_model) -> (batch,seq_len,vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)  

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder , src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)    

    def project(self,x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int =512, N: int =6, h: int =8, dropout: float = 0.1,  d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers 
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

        # create the decoder blocks 
    decoder_blocks = [] 
    for _ in range(n):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)      
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)   
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters with Glorot initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer