# Questions & Answers

## Q1. Autoencoder란 무엇인가요 ? 오토인코더의 다양한 층에 대해 설명하고, 오토인코더의 세 가지 실용적인 사용 사례를 언급해주세요

A. Autoencoder는 비지도학습(Unsupervised Learning)에 사용되는 딥러닝 알고리즘 중 하나입니다. 오토인코더에는 input layer, Encoder, bottleneck latyer, decoder, output layer 라는 중요한 특징이 있습니다. 

Encoder는 input data를 인코딩된 표현으로 압축하며, 이 인코딩된 표현은 일반적으로 입력 데이터보다 훨씬 작습니다.

Latent Space Representations(잠재 공간 표현)/ Bottleneck (병목) / Code - 입력값의 가장 중요한 특징을 포함하는 압축된 요약본 입니다. 

Decoder - 지식 표현을 Decompress 하고 인코딩된 형태로부터 데이터를 복원합니다. 그런 다음 손실 함수가 맨 위에서 입력 이미지와 출력 이미지를 비교하는 데 사용됩니다. 참고로 입력과 출력의 차원이 동일해야 한다는 요구 사항이 있습니다. 중간에 있는 모든 것들은 자유롭게 조정할 수 있습니다.

오토인코더는 실생활에서 매우 다양하게 사용됩니다. 

- 트랜스포머 모델과 Big Bird (오토인코더는 두 알고리즘의 요소들 중 하나입니다.) : 텍스트 축약, 텍스트 생성 등 
- 이미지 압축 
- PCA의 비선형 버전