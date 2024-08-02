# Positional Encoding

(1)

nn.Embedding(vocab_size, d_model)은 (vocab_size, d_model) 크기의 가중치 행렬을 생성합니다.

- 각 행은 하나의 토큰에 대한 d_model 차원의 임베딩 벡터입니다. : 행뽑기다

입력으로 정수 인덱스(0에서 vocab_size-1 사이의 값)가 주어지면, 해당 인덱스에 해당하는 행(벡터)을 반환합니다.



