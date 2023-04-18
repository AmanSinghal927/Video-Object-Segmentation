import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import VideoFramesDataset


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, args):
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, train_acc *
            len(train_loader.dataset), len(train_loader.dataset),
            100. * train_acc))

        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            val_loss, val_acc *
            len(val_loader.dataset), len(val_loader.dataset),
            100. * val_acc))

        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            if args.save_model:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                torch.save(model.state_dict(), os.path.join(
                    args.save_dir, 'model.pth'))
                
        print('Best accuracy: {:.0f}%'.format(100. * best_acc))

def unsupervised_train(model, unlabel_loader, optimizer, criterion, args):
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(unlabel_loader):
            data = data.to(args.device)
            optimizer.zero_grad()

            # TODO: figure out how to do unsupervised training
            # What is input?
            #   - a batch is a list of videos (each video is a list of frame images)
            #   - each video is a list of frame images
            #   - each frame image is a tensor of shape (3, 160, 240)
            #   - each batch is a tensor of shape (batch_size, num_frames?, 3, 160, 240)
            # What is target?

            output = model(data)
            loss = criterion(output, output)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Unsupervised Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(unlabel_loader.dataset),
                    100. * batch_idx / len(unlabel_loader), loss.item()))
                
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='model')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--debug_dataloader', action='store_true', default=False)
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load data
    train_loader = DataLoader(
        dataset=VideoFramesDataset(args.data_dir, 'train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    unlabel_loader = DataLoader(
        dataset=VideoFramesDataset(args.data_dir, 'unlabel'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=VideoFramesDataset(args.data_dir, 'val'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.debug_dataloader:
        for batch_idx, data in enumerate(train_loader):
            print(data.shape)
            break
    exit()

    # Load model
    model = ...  # TODO: load model

    # Load optimizer
    optimizer = ...  # TODO: choose optimizer

    # Load scheduler
    scheduler = ...  # TODO: choose scheduler

    # Load loss function
    criterion = ...  # TODO: choose loss function

    # Load checkpoint
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

    # Train model
    if args.unsupervised:
        unsupervised_train(model, unlabel_loader, optimizer, criterion, args)
    else:
        train(model, train_loader, val_loader,
            optimizer, scheduler, criterion, args)

    # Save model
    if args.save_model:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'))
