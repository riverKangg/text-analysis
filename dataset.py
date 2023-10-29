import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, input_tensors, attention_tensors, label_tensors):
        self.input_tensors = input_tensors
        self.attention_tensors = attention_tensors
        self.label_tensors = label_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, index):
        input_data = self.input_tensors[index]
        attention_data = self.attention_tensors[index]
        label_data = self.label_tensors[index]
        return input_data, attention_data, label_data

# 예시로 두 개의 텐서를 생성
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 입력 데이터 텐서
label_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]])  # 레이블 텐서

# 사용자 정의 데이터셋 객체 생성
custom_dataset = CustomDataset(input_tensor, label_tensor)

# 데이터셋을 DataLoader로 변환하여 미니배치로 로드할 수 있습니다.
from torch.utils.data import DataLoader

batch_size = 64  # 배치 크기 설정
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# data_loader를 사용하여 데이터를 반복적으로 가져올 수 있음
for batch in data_loader:
    input_data, label_data = batch
    # 이곳에서 모델에 데이터를 입력하거나 원하는 작업 수행
