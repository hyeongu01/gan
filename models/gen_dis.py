import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_size=[512, 512, 512], output_size=784):
        super(Generator, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        self.hidden_layers.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_size[0]),
                nn.BatchNorm1d(hidden_size[0]),
                nn.ReLU()
            )
        )
        
        for i in range(len(hidden_size) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size[i], hidden_size[i+1]),
                    nn.BatchNorm1d(hidden_size[i+1]),
                    nn.ReLU()
                )
            )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size[-1], output_size),
            nn.Tanh()
        )
            
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
# maxout class
class Maxout(nn.Module):
    def __init__(self, input_size, hidden_size, size_units=2, num_mo_layers=3, dropout_prob=0.5):
        super(Maxout, self).__init__()
        
        self.mo_layers = nn.ModuleList()
        
        for i in range(num_mo_layers):
            self.mo_layers.append(nn.Sequential(
                nn.Linear(input_size, hidden_size*size_units),
                nn.BatchNorm1d(hidden_size*size_units)
            ))
        self.size_units = size_units
        self.num_mo_layers = num_mo_layers
        
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        ini_x = [layer(x) for layer in self.mo_layers]
        for i in range(self.num_mo_layers):
            ini = ini_x[i].view(ini_x[i].size(0), -1, self.size_units)
            ini_x[i], _ = ini.max(dim=2)
            ini_x[i] = self.dropout(ini_x[i])
        output = torch.cat(ini_x, dim=1)
        
        return output
    
# Discriminator with Maxout class
class Discriminator(nn.Module):
    def __init__(self, input_size=784, hidden_size=[512, 256, 128], num_mo_layer=3, output_size=1, dropout_prob=0.5):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.ReLU()
        )
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_size) - 1):
            if i == 0:
                in_size = hidden_size[i]
            else:
                in_size = hidden_size[i] * 3
            self.hidden_layers.append(
                Maxout(input_size=in_size, hidden_size=hidden_size[i+1], dropout_prob=dropout_prob)
            )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size[-1]*num_mo_layer, output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
def build_generator(latent_dim=100, hidden_size=[512, 512, 512], output_size=784):
    gen = Generator(
        latent_dim=latent_dim,
        hidden_size=hidden_size,
        output_size=output_size
    )
    return gen

def build_discriminator(input_size=784, hidden_size=[512, 256, 128], num_mo_layer=3, output_size=1):
    dis = Discriminator(
        input_size=input_size,
        hidden_size=hidden_size,
        num_mo_layer=num_mo_layer,
        output_size=output_size
    )
    return dis
    

if __name__ == "__main__":
    gen = Generator()
    print(f" - Generator model structure\n{gen}")
    
    z = torch.rand(64, 100) * 2 - 1  # [0, 1] → [-1, 1]

    # 확인
    print(" - Generator input data")
    print("size:", z.shape)
    print("최소값:", z.min().item())
    print("최대값:", z.max().item())

    output = gen(z)
    print(" - Generator output data")
    print("size:", output.shape)
    print("최소값:", output.min().item())
    print("최대값:", output.max().item())
    
    # maxout class test
    maxout = Maxout(512, 256)
    print(f"\n - Maxout class structure\n{maxout}")

    input_tensor = torch.rand(64, 512) # [64, 512]

    output = maxout(input_tensor)
    print(" -- Maxout class test. --")
    print(f"input: {input_tensor.shape}\noutput: {output.shape}")
    
    # Discriminator test    
    dis = Discriminator()
    print(f"\n - Discriminator model structure\n{dis}")

    input_tensor = torch.rand(64, 784)
    output_tensor = dis(input_tensor)
    print(" -- Discriminator test -- ")
    print(f"input size: {input_tensor.shape}\noutput size: {output_tensor.shape}")
    
    