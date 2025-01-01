import argparse
import os
from data.datasets import MNISTDataset, transform
from models.gan import Gan
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Experiment for GAN study")
    
    # essential parameters
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for training")
    parser.add_argument("--early_epoch", type=int, default=3, help="Early epoch")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training")
    
    # set save path
    parser.add_argument("--model_path", type=str, default="checkpoints", help="Path for model checkpoint")
    parser.add_argument("--tensorboard_path", type=str, default="experiment_01", help="Path for tensorboard logging")
    
    
    # additional parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--k", type=int, default=1, help="In Gan training number of training Discriminator")
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer, select 'SGD' or 'Adam'")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="In Adam optimizer, set 'weight_decay'")
    
    return parser.parse_args()


def main(args):
    # load trainLoader in mnist
    mnist_data_path = "./mnist"
    if os.path.exists(mnist_data_path):
        train_dataset = MNISTDataset(mnist_data_path, train=False, transform=transform, download=False)
    else:
        train_dataset = MNISTDataset(mnist_data_path, train=True, transform=transform, download=True)
    
    trainLoader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # load Gan model
    gan = Gan(k=args.k, device=args.device)
    
    # set optimizer
    gan.set_optimizer(optim_name=args.optim, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # train
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    gan.train(
        trainLoader=trainLoader,
        epochs=args.epochs,
        log_path=os.path.join("runs", args.tensorboard_path),
        early_epoch=args.early_epoch,
        checkpoint_dir=args.model_path
    )

if __name__ == "__main__":
    args = get_args()
    
    main(args)