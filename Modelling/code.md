# Torch 문법 

### nn.Parameter

```python
# 일반 텐서
self.tensor = torch.randn(3, 4)  # 모델 파라미터로 인식되지 않음

# nn.Parameter
self.parameter = nn.Parameter(torch.randn(3, 4))  # 모델 파라미터로 인식됨

self.alpha = nn.Parameter(torch.ones(d_model))
self.bias = nn.Parameter(torch.zeros(d_model))
```

`Layer Normalization` 클래스 구현 중

`alpha` 와 `bias`를 학습 가능한 파라미터로 정의합니다. 백프로파게이션 동안 업데이트됩니다. Optimizer가 자동으로 이 파라미터를 인식하고 업데이트 합니다. 

`nn.Parameter`를 사용함으로써, PyTorch에게 "이 텐서는 모델의 중요한 부분이며, 학습 과정에서 업데이트되어야 한다"고 알려주는 것입니다. Layer Normalization에서 alpha와 bias를 nn.Parameter로 정의함으로써, 이들이 학습 과정에서 조정되어 모델의 성능을 향상시킬 수 있게 됩니다.