import argparse
from models.gan import Gan
from data.datasets import denorm
import torch
import os
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Experiment for GAN study")
    
    # essential parameters
    parser.add_argument("--device", type=str, default="cpu", help="Device for training")
    parser.add_argument("--number_test", type=int, default=50, help="Generating number")
    
    # set load & save path
    parser.add_argument("--model_path", type=str, default="checkpoints/epoch_32.pth", help="Path for model checkpoint")
    parser.add_argument("--output_path", type=str, default="outputs", help="Output path")

    return parser.parse_args()

def main(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    gan = Gan(device=args.device)
    gan.load_checkpoint(args.model_path)
    
    z = torch.randn(args.number_test, 100).to(args.device)
    
    gan.generator.eval()
    outputs = gan.generator(z)
    print(outputs.shape)
    for i in range(args.number_test):
        image = outputs[i]
        img = denorm(image).view(28, 28)
        img = img.detach().numpy()
        image_path = os.path.join(args.output_path, f"{i}.png")
        plt.imsave(image_path, img, cmap="gray")
    
if __name__ == "__main__":
    args = get_args()
    main(args)