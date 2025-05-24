from model import MRUNet
from utils import *
from tiff_process import tiff_process
from dataset import LOADDataset
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MR UNet training from tif files contained in a data folder",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', help='path to directory containing training tif data')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=24, type=int, help='size of batch')
    parser.add_argument('--model_name', type=str, help='name of the model')
    parser.add_argument('--continue_train', choices=['True', 'False'], default='False', type=str,
                        help="flag for continue training, if True - continue training the 'model_name' model, else - training from scratch")
    return parser.parse_args()

def train(model, dataloader, optimizer, train_data, max_val):
    model.train()
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        image_data, label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = get_loss(outputs * max_val, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_psnr += psnr(label, outputs, max_val)
        running_ssim += ssim(label, outputs, max_val)
    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(len(train_data)/dataloader.batch_size)
    final_ssim = running_ssim / int(len(train_data)/dataloader.batch_size)
    return final_loss, final_psnr, final_ssim

def validate(model, dataloader, epoch, val_data, max_val):
    model.eval()
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            image_data, label = data[0].to(device), data[1].to(device)
            outputs = model(image_data)
            loss = get_loss(outputs * max_val, label)
            running_loss += loss.item()
            running_psnr += psnr(label, outputs, max_val)
            running_ssim += ssim(label, outputs, max_val)
    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(len(val_data)/dataloader.batch_size)
    final_ssim = running_ssim / int(len(val_data)/dataloader.batch_size)
    return final_loss, final_psnr, final_ssim

def main():
    global device
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MRUNet(res_down=True, n_resblocks=1, bilinear=0).to(device)

    scale = 4

    # Tiff process
    Y_day, Y_night = tiff_process(args.datapath)
    Y = np.concatenate((Y_day, Y_night), axis=0)

    # Remove all-zero images
    Y_new = [img for img in Y if len(img[img == 0]) <= 0]
    Y_new = np.array(Y_new)
    start = time.time()

    np.random.seed(1)
    np.random.shuffle(Y_new)
    ratio = 0.75
    y_train = Y_new[:int(Y_new.shape[0]*ratio)]
    y_val = Y_new[int(Y_new.shape[0]*ratio):]

    # Data augmentation (flipping)
    y_train_new = []
    for img in y_train:
        y_train_new.append(img)
        y_train_new.append(np.flip(img, 1))
    y_train = np.array(y_train_new)
    max_val = np.max(y_train)
    print(f'Max pixel value of training set is {max_val},\nIMPORTANT: Please save it for later use as the normalization factor\n')

    # Create inputs with bicubic preprocessing
    x_train = np.zeros_like(y_train)
    for i in range(y_train.shape[0]):
        a = downsampling(y_train[i], scale)
        x_train[i] = normalization(upsampling(a, scale), max_val)

    x_val = np.zeros_like(y_val)
    for i in range(y_val.shape[0]):
        a = downsampling(y_val[i], scale)
        x_val[i] = normalization(upsampling(a, scale), max_val)

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1], x_val.shape[2]))
    y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1], y_train.shape[2]))
    y_val = y_val.reshape((y_val.shape[0], 1, y_val.shape[1], y_val.shape[2]))
    end = time.time()
    print(f"Finished processing data in additional {(end - start)/60:.3f} minutes \n")

    transform = None
    train_data = LOADDataset(x_train, y_train, transform=transform)
    val_data = LOADDataset(x_val, y_val, transform=transform)
    batch_size = args.batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    print(f'Length of training set: {len(train_data)}')
    print(f'Length of validating set: {len(val_data)}')
    print(f'Shape of input and output: ({x_train.shape[-2]},{x_train.shape[-1]})')

    epochs, lr = args.epochs, args.lr
    model_name = args.model_name
    continue_train = args.continue_train == 'True'
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists("Metrics"):
        os.makedirs("Metrics")

    if not continue_train:
        train_loss, val_loss = [], []
        train_psnr, val_psnr = [], []
        train_ssim, val_ssim = [], []
        start = time.time()
        last_epoch = -1
        vloss = np.inf
    else:
        metrics = np.load(os.path.join("./Metrics", model_name + ".npy"))
        train_loss, val_loss = metrics[0].tolist(), metrics[3].tolist()
        train_psnr, val_psnr = metrics[1].tolist(), metrics[4].tolist()
        train_ssim, val_ssim = metrics[2].tolist(), metrics[5].tolist()
        start = time.time()
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        vloss = checkpoint['losses'][3]

    for epoch in range(last_epoch + 1, epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_psnr, train_epoch_ssim = train(model, train_loader, optimizer, train_data, max_val)
        val_epoch_loss, val_epoch_psnr, val_epoch_ssim = validate(model, val_loader, epoch, val_data, max_val)
        print(f"Train loss: {train_epoch_loss:.6f}")
        print(f"Val loss: {val_epoch_loss:.6f}")

        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        train_ssim.append(train_epoch_ssim)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
        val_ssim.append(val_epoch_ssim)

        if val_epoch_loss < vloss:
            print("Save model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': [train_epoch_loss, train_epoch_psnr, train_epoch_ssim,
                           val_epoch_loss, val_epoch_psnr, val_epoch_ssim],
            }, model_name)
            np.save(os.path.join("./Metrics", model_name),
                    [train_loss, train_psnr, train_ssim, val_loss, val_psnr, val_ssim])
            vloss = val_epoch_loss
    end = time.time()
    print(f"Finished training in: {(end - start)/60:.3f} minutes")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
