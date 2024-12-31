from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([
    transforms.ToTensor(), # [0, 255] -> 0에서 1값으로 정규화
    transforms.Normalize((0.5, ), (0.5, )) # 0에서 1값으로 정규화
])

class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.data = datasets.MNIST(root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample, _ = self.data[index]
        
        
        return torch.flatten(sample)
    

def denorm(tensor, mean=0.5, std=0.5):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    tensor = (tensor * std + mean) * 255
    tensor = tensor.round().clamp(0, 255).byte() # round(): 반올림 처리
    
    return tensor

if __name__ == "__main__":
    train_dataset = MNISTDataset("../mnist", train=True, transform=transform, download=False)
    test_dataset = MNISTDataset("../mnist", train=False, transform=transform, download=False)
    
    trainLoader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )
    testLoader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )
    for batch_idx, images in enumerate(trainLoader):
        print(f"배치 인덱스: {batch_idx}")
        print(f"이미지 크기: {images.shape}")
        print("최대값:", images.min().data)
        print("최소값:", images.max().data)
        break