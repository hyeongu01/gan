import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from gen_dis import build_discriminator, build_generator

# for test
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

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

class Gen_loss(nn.Module):
    def __init__(self, eps=1e-8):
        super(Gen_loss, self).__init__()    
        self.is_early = True
        self.eps=eps
        
    def set_phase(self, is_early):
        self.is_early = is_early
        
    def forward(self, d_gz):
        if self.is_early:
            loss = -torch.log(d_gz + self.eps).mean()
        else:
            loss = torch.log(1 - d_gz + self.eps).mean()
            
        return loss
    
class Dis_loss(nn.Module):
    def __init__(self, eps=1e-8):
        super(Dis_loss, self).__init__()
        self.eps=eps
        
        
    def forward(self, d_x, d_gz):
        x_loss = -torch.log(d_x + self.eps).mean()
        z_loss = -torch.log(1 - d_gz + self.eps).mean()
        loss = x_loss + z_loss
        return loss
    


class Gan(nn.Module):
    def __init__(self, k=1, device="cpu"):
        super(Gan, self).__init__()
        
        self.generator = build_generator().to(device)
        self.discriminator = build_discriminator().to(device)
        
        self.k = k
        self.device = device
        
        self.gen_loss = Gen_loss()
        self.dis_loss = Dis_loss()
        
        self.optim_g = torch.optim.SGD(self.generator.parameters(), lr=0.0001, momentum=0.9)
        self.optim_d = torch.optim.SGD(self.discriminator.parameters(), lr=0.0001, momentum=0.9)
        
    
    def set_loss(self, gen_loss, dis_loss):
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss
    
    def set_optimizer(self, optimizer_g, optimizer_d):
        self.optim_g = optimizer_g
        self.optim_d = optimizer_d
        
    def train_one_epoch(self, dataLoader, epoch=0):
        # train discriminator k steps
        self.discriminator.train()
        self.generator.eval()
        
        epoch_loss_d = 0.0
        for k in range(self.k):
            with tqdm(dataLoader, unit="batch", leave=False) as tepoch:
                for x in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1} | discriminator {k+1}/{self.k}")

                    self.optim_d.zero_grad()
                    
                    x = x.to(self.device)
                    
                    z = torch.randn(64, 100).to(self.device)
                    gz = self.generator(z)
                    d_gz = self.discriminator(gz)
                    dx = self.discriminator(x)
                    
                    loss_d = self.dis_loss(dx, d_gz)
                    loss_d.backward()
                    
                    self.optim_d.step()
                    epoch_loss_d += loss_d.item()

        epoch_loss_d = epoch_loss_d / self.k

        
        # train generator one steps
        self.generator.train()
        self.discriminator.eval()
        
        epoch_loss_g = 0.0
        with tqdm(dataLoader, unit="batch", leave=False) as tepoch:
            for x in tepoch:
                tepoch.set_description(f"Epoch {epoch+1} | generator")

                self.optim_g.zero_grad()
                x = x.to(self.device)
                
                z = torch.randn(64, 100).to(self.device)
                gz = self.generator(z)
                d_gz = self.discriminator(gz)
                
                loss_g = self.gen_loss(d_gz)
                loss_g.backward()
                
                self.optim_g.step()
                epoch_loss_g += loss_g.item()
        
        return epoch_loss_d, epoch_loss_g
    
    def train(self, trainLoader, epochs, log_path="experiment_01", early_rate=0.1):
        writer = SummaryWriter(log_dir=f"./runs/{log_path}")

        z = torch.randn(3, 100).to(self.device)
        for batch_x in dataloader:
            sample_data = batch_x[:3]
            sample_data = sample_data.view(3, 1, 28, 28)
            sample_data = (sample_data + 1) / 2
            grid = make_grid(sample_data, nrow=3, padding=2)
            writer.add_image("Origin Images", grid, 0)
            break

        early_epoch = int(epochs * early_rate)
        for epoch in range(epochs):
            if epoch > early_epoch:
                self.gen_loss.set_phase(False)
                
            train_loss_d, train_loss_g = self.train_one_epoch(trainLoader, epoch=epoch)
            writer.add_scalar("Loss/Discriminator", train_loss_d, epoch)
            writer.add_scalar("Loss/Generator", train_loss_g, epoch)

            if epoch % 2 == 0:
                test_output = self.generator(z)
                test_output = test_output.view(3, 1, 28, 28)
                test_output = (test_output + 1) / 2

                grid = make_grid(test_output, nrow=3, padding=2)
                writer.add_image("Output Images", grid, epoch)

            print(f"Epoch [{epoch+1}/{epochs}] train_loss_d: {train_loss_d}, train_loss_g: {train_loss_g}")
        writer.close()

        

    def forward(self, x):
        return x
    
if __name__ == "__main__":
    train_dataset = MNISTDataset('./mnist', train=True, transform=transform)

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    gan = Gan(device="cuda:0", k=3)
    gan.train(trainLoader=dataloader, log_path="ex02", epochs=100)