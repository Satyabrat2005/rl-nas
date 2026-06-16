import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import wandb
import os

from search_space import Cell, OPS_NAMES

class SupernetFixed(nn.Module):
    def __init__(self, C_init=16, n_cells=10, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_init, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_init),
            nn.ReLU(inplace=True)
        )

        cells = []
        adapters = []
        channel_projectors = []
        curr_C = C_init
        for i in range(n_cells):
            reduction = (i == 3 or i == 6)
            if reduction:
                new_C = curr_C * 2
                adapters.append(nn.Conv2d(curr_C, new_C, 1, stride=2, bias=False))
                curr_C = new_C
            else:
                adapters.append(nn.Identity())
            cells.append(Cell(curr_C, reduction=reduction))
            channel_projectors.append(nn.Conv2d(2 * curr_C, curr_C, 1, bias=False))

        self.cells = nn.ModuleList(cells)
        self.adapters = nn.ModuleList(adapters)
        self.channel_projectors = nn.ModuleList(channel_projectors)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(curr_C, n_classes)

    def forward(self, x):
        s = self.stem(x)
        s0 = s1 = s
        for i, cell in enumerate(self.cells):
            s0 = self.adapters[i](s0)
            s1 = self.adapters[i](s1)
            # Uniform edge weights (zero logits -> uniform softmax)
            edge_weights = {}
            for node in range(2, 6):
                for edge_idx in range(2):
                    key = f"node{node}_edge{edge_idx}"
                    edge_weights[key] = torch.zeros(len(OPS_NAMES), device=x.device)
            out = cell(s0, s1, edge_weights=edge_weights)
            out = self.channel_projectors[i](out)
            s0, s1 = s1, out
        out = self.global_pool(s1).flatten(1)
        return self.classifier(out)


def get_cifar10_loaders(batch_size=96):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def train_fixed(n_epochs=200, device='cuda', C_init=16):
    model = SupernetFixed(C_init=C_init, n_cells=10).to(device)
    train_loader, val_loader = get_cifar10_loaders(batch_size=96)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    wandb.init(project='rl-nas', name='fixed-10cells', config={'epochs': n_epochs, 'C_init': C_init})

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    logits = model(images)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100.0 * correct / total
        scheduler.step()
        wandb.log({'epoch': epoch+1, 'loss': avg_loss, 'val_acc': val_acc})
        print(f"Epoch {epoch+1}/{n_epochs} | loss:{avg_loss:.4f} | val_acc:{val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/fixed_best.pth')
    wandb.finish()
    return best_acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Training fixed-architecture supernet (10 cells, 200 epochs)...")
    best = train_fixed(n_epochs=200, device=device)
    print(f"Best validation accuracy: {best:.2f}%")
