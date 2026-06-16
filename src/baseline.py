import torch 
import torchvision 
import torchvision.transforms as transforms
from torch import device, device, nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from rich import print

#config.
config = {
    "batch_size": 96,
    "num_epochs": 200,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "cos_lr": True,
    "save_interval": 5,
    "num_workers": 4,
    "amp": True
}

#ResNet-20 model
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])

#Data Preparation 
transform_train = transforms.Compose([ 
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(project="rl-nas", name="baseline-resnet20", config=config)

    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'], weight_decay=config['weight_decay'])
    if config['cos_lr']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    scaler = GradScaler(enabled=config['amp'])
    best_val_acc = 0.0
    checkpoint_path = 'checkpoints/baseline_best.pth'

    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast(enabled=config['amp']):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)

        # Step scheduler once per epoch
        scheduler.step()

        # Validation
        val_acc = evaluate(model, testloader)
        avg_train_loss = running_loss / len(trainset)

        wandb.log({
            'train_loss': avg_train_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        if (epoch + 1) % config['save_interval'] == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with val acc: {val_acc:.2f}%")

    # Final test evaluation
    model.load_state_dict(torch.load(checkpoint_path))
    test_acc = evaluate(model, testloader)
    print(f"Final test accuracy of best model: {test_acc:.2f}%")
    wandb.finish()

if __name__ == '__main__':
    main()
