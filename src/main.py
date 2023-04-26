import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FrameDataset
from model.jepa import JEPA
from model.encoder import ViViT, HierarchicalAttentionEncoder, FeedForward, Transformer
from torchvision import transforms
from x_transformers import Decoder, Encoder
import copy

JEPAs = [{
    "model": None,
    "optimizer": None,
    "scheduler": None,
    "criterion": None,
    "args": None
} for _ in range(11)]

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, args):
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)

            output = model(data)
            
            optimizer.zero_grad()
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
                    args.save_dir, f'model_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth'))
                
        print('Best accuracy: {:.0f}%'.format(100. * best_acc))

def unsupervised_train(model, unlabel_loader, optimizer, criterion, scheduler, args, skip=0):
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
             # split the frames into x and y
            # x: (b, 11, c, h, w)
            # y: (b, 1, c, h, w)
            data_x = data[:, :11, :, :, :]
            data_y = data[:, 11 + skip, :, :, :].unsqueeze(1)
   
            # print("x: ", x.shape)
            # print("y: ", y.shape)

            # rearrange for encoder (b, num_frames=22, c, h, w) to (b, c, num_frames=22, h, w)
            data_x = data_x.permute(0, 2, 1, 3, 4)
            data_y = data_y.permute(0, 2, 1, 3, 4)

            # encode with previous JEPAs
            encoded_xs = []
            encoded_ys = []
            with torch.no_grad():
                for i in range(skip):
                    enc_x = JEPAs[i]["model"].encoder_x(data_x)
                    enc_y = JEPAs[i]["model"].encoder_y(data_y)
                    encoded_xs.append(enc_x)
                    encoded_ys.append(enc_y)

            x, target, latent_loss = model(data_x, data_y, encoded_xs, encoded_ys)

            loss = criterion(x, target) + (args.lamb * latent_loss)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Unsupervised Train Epoch for JEPA #{} : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    skip + 1, epoch, batch_idx * len(data), len(unlabel_loader.dataset),
                    100. * batch_idx / len(unlabel_loader), loss.item()))  
        scheduler.step()
                
    # save model
    if args.save_model:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'pretrain_JEPA_skips{skip}.pth'))

def finetune_VAE(model, train_loader, val_loader, optimizer, criterion, scheduler, args):
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            data = data.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(data.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Finetune VAE Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        print('Finetune VAE Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, train_acc *
            len(train_loader.dataset), len(train_loader.dataset),
            100. * train_acc))

        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(args.device)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(data.view_as(pred)).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        print('Finetune VAE Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
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
                    args.save
                    _dir, f'finetune_VAE_{args.model}_best.pth'))

    if args.save_model:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'finetune_VAE_{args.model}_last.pth'))
                      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='model')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--frame_skip', type=int, default=0)
    parser.add_argument('--debug_dataloader', action='store_true', default=False)
    parser.add_argument('--lamb', type=float, default=0.1)
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check which device is available
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")


    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load data
    train_loader = DataLoader(
        dataset=FrameDataset(args.data_dir + "/" + 'train', labeled=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=FrameDataset(args.data_dir + "/" + 'val', labeled=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.unsupervised:
        ds = FrameDataset(args.data_dir + "/" + 'unlabeled', labeled=False, transform=transform)
        # ds = torch.utils.data.Subset(ds, list(range(0, 100)))
        unlabel_loader = DataLoader(
            dataset=ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    if args.debug_dataloader:
        for batch_idx, data in enumerate(train_loader):
            frames, mask = data
            print(frames.shape)
            print(mask.shape)
            break

    # Layers
    encoder_x = ViViT(
        image_size=(160, 240),
        image_patch_size=(8, 8),
        frames=11,
        frame_patch_size=1,
        num_classes=512,
        dim=512,
        spatial_depth=6,
        temporal_depth=6,
        heads=8,
        mlp_dim=2048,
    )

    encoder_y = ViViT(
        image_size=(160, 240),
        image_patch_size=(8, 8),
        frames=1,
        frame_patch_size=1,
        num_classes=512,
        dim=512,
        spatial_depth=6,
        temporal_depth=6,
        heads=8,
        mlp_dim=2048,
    )

    # predictor, output (bs, embed_dim)
    predictor = Decoder(
        dim=512,
        depth=6,
        heads=8,
    )

    enc = Encoder(
        dim=512,
        depth=6,
        heads=8,
    )

    rand_frames = torch.rand((args.batch_size, 3, 11, 160, 240))

    try:
        enc_rand = encoder_x(rand_frames)
        print(enc_rand.shape)
        pred_rand = predictor(enc_rand)
        print(pred_rand.shape)
        enc_rand = enc(enc_rand)
        print(enc_rand.shape)
    except:
        print("Error in model")

    # encoder predictor module (bs, 512) -> (bs, 512)
    # 
    predictor = Transformer(
        dim=512,
        depth=6,
        heads=8,
        dim_head=64, 
        mlp_dim=2048, 
        dropout=0.1,
    )

    # Load model
    # model = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
    #              embed_dim=512, 
    #              encoder_x=encoder_x, 
    #              encoder_y=encoder_y, 
    #              predictor=predictor, 
    #              skip=args.frame_skip
    #              ).to(args.device)

    # # Load optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # # Load scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # # Load loss function
    # criterion = nn.MSELoss()

    # for i in range(len(JEPAs)):
    #     m = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
    #                 embed_dim=512, 
    #                 encoder_x=copy.deepcopy(encoder_x), 
    #                 encoder_y=copy.deepcopy(encoder_y), 
    #                 predictor=copy.deepcopy(predictor), 
    #                 skip=i
    #                 ).to(args.device)
    #     JEPAs[i]["model"] = m
    #     JEPAs[i]["optimizer"] = optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)
    #     JEPAs[i]["scheduler"] = optim.lr_scheduler.StepLR(JEPAs[i]["optimizer"], step_size=10, gamma=0.1)
    #     JEPAs[i]["criterion"] = nn.MSELoss()

    print("Models loaded")

    # Load pretrained model
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))

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
        # unsupervised_train(model, unlabel_loader, optimizer, criterion, scheduler, args)
        for i in range(len(JEPAs)):
            hsa_x = HierarchicalAttentionEncoder(
                num_encoders=i + 1,
                embed_dim=512,
                hidden_dim=512,
            )
            hsa_y = copy.deepcopy(hsa_x)
            m = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
                    embed_dim=512, 
                    encoder_x=copy.deepcopy(encoder_x), 
                    encoder_y=copy.deepcopy(encoder_y), 
                    hsa_x=hsa_x,
                    hsa_y=hsa_y,
                    predictor=copy.deepcopy(predictor), 
                    skip=i
                    ).to(args.device)
            JEPAs[i]["model"] = m
            JEPAs[i]["optimizer"] = optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)
            JEPAs[i]["scheduler"] = optim.lr_scheduler.StepLR(JEPAs[i]["optimizer"], step_size=10, gamma=0.1)
            JEPAs[i]["criterion"] = nn.MSELoss()  
            print(f"Training JEPA {i}")
            unsupervised_train(JEPAs[i]["model"], unlabel_loader, JEPAs[i]["optimizer"], JEPAs[i]["criterion"], JEPAs[i]["scheduler"], args, i)
    else:
        train(model, train_loader, val_loader,
            optimizer, scheduler, criterion, args)

    # Save model
    if args.save_model and not args.unsupervised:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'))
