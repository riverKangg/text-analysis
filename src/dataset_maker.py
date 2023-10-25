import torch
import torch.utils.data as data


class BasicDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(BasicDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    train_x = torch.rand(500)
    train_y = torch.rand(500)
    tr_dataset = BasicDataset(train_x, train_y)