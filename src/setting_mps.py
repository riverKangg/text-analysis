import torch


mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()
print(f'MPS 사용 가능 여부: {mps_available}')
print(f'MPS 빌드 여부: {mps_built}')



import torch
import torch.nn as nn

# MPS 사용 설정
dtype = torch.float
device = torch.device("mps")

# 랜덤 데이터 생성
x = torch.rand(100, 3, device=device, dtype=dtype)
# y 값은 다항식 형태로 생성
y = 3 * x[:, 0] + 2 * x[:, 1] - x[:, 2] + 1 + 0.1 * torch.randn(100, device=device, dtype=dtype)

# 선형 모델 구성
ml_model = nn.Linear(3, 1)  # 3x1 (변수 할당)
ml_model.to(device)

# 옵티마이저
optimizer = torch.optim.SGD(ml_model.parameters(), lr=0.01)

# epoch 설정
total_epoch = 10000

# 학습 시작
for epoch in range(total_epoch + 1):
    # 예측값
    prediction = ml_model(x)

    # 비용 (평균 제곱 오차 사용)
    loss = nn.functional.mse_loss(prediction, y)

    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 중간 기록
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{total_epoch}, Loss: {loss.item()}")

# 학습된 모델의 가중치 출력
print("학습된 모델의 가중치 (w1, w2, w3):", ml_model.weight.data)
print("학습된 모델의 편향 (b):", ml_model.bias.data)