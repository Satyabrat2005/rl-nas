import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import wandb
import os
import json
from typing import List, Tuple

from search_space import OPS_NAMES, Cell

# Discrete Evaluator (full retraining for proxy epochs)
class DiscreteEvaluator(nn.Module):
    """Same architecture as fixed supernet, but with discrete ops."""
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

    def set_discrete_architecture(self, arch_ops: List[int]):
        idx = 0
        for cell in self.cells:
            edge_weights = {}
            for node in range(2, 6):
                for edge_idx in range(2):
                    key = f"node{node}_edge{edge_idx}"
                    one_hot = torch.zeros(len(OPS_NAMES))
                    one_hot[arch_ops[idx]] = 1.0
                    edge_weights[key] = one_hot
                    idx += 1
            if not hasattr(cell, '_original_forward'):
                cell._original_forward = cell.forward
            def make_new_forward(cell):
                def new_forward(self, s0, s1, edge_weights=None):
                    return cell._original_forward(s0, s1, edge_weights=cell._fixed_weights)
                return new_forward
            cell._fixed_weights = edge_weights
            cell.forward = make_new_forward(cell).__get__(cell, Cell)
        return

    def forward(self, x):
        s = self.stem(x)
        s0 = s1 = s
        for i, cell in enumerate(self.cells):
            s0 = self.adapters[i](s0)
            s1 = self.adapters[i](s1)
            out = cell(s0, s1)
            out = self.channel_projectors[i](out)
            s0, s1 = s1, out
        out = self.global_pool(s1).flatten(1)
        return self.classifier(out)

# Data loaders (reused for each architecture training)
def get_cifar10_loaders(batch_size=96):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

# Retraining evaluator (10 proxy epochs)
def evaluate_architecture(arch_ops: List[int], device: torch.device,
                          n_epochs: int = 10, batch_size: int = 96) -> float:
    print(f"[Evaluator] Training architecture {arch_ops[:5]}... for {n_epochs} epochs")
    model = DiscreteEvaluator(C_init=16, n_cells=10).to(device)
    model.set_discrete_architecture(arch_ops)

    train_loader, val_loader = get_cifar10_loaders(batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total
    print(f"[Evaluator] Accuracy: {acc:.2f}%")
    return acc

# REINFORCE Controller (LSTM)
class Controller(nn.Module):
    def __init__(self, n_ops=7, n_cells=10, lstm_hidden=64):
        super().__init__()
        self.n_ops = n_ops
        self.n_cells = n_cells
        self.num_edges_per_cell = (6-2)*2
        self.total_edges = n_cells * self.num_edges_per_cell
        self.op_embedding = nn.Embedding(n_ops, 32)
        self.lstm = nn.LSTM(32, lstm_hidden, num_layers=2, batch_first=True)
        self.op_head = nn.Linear(lstm_hidden, n_ops)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

    def forward(self, actions, hidden):
        emb = self.op_embedding(actions)
        out, new_hidden = self.lstm(emb, hidden)
        logits = self.op_head(out[:, -1, :])
        return logits, new_hidden

    def sample_architecture(self, device):
        self.train()
        actions = []
        log_probs = []
        hidden = (torch.zeros(2, 1, self.lstm.hidden_size, device=device),
                  torch.zeros(2, 1, self.lstm.hidden_size, device=device))
        for t in range(self.total_edges):
            if t == 0:
                act = torch.zeros(1, 1, dtype=torch.long, device=device)
            else:
                act = torch.tensor([[actions[-1]]], dtype=torch.long, device=device)
            logits, hidden = self.forward(act, hidden)
            probs = F.softmax(logits, dim=-1)
            a = torch.multinomial(probs, 1).item()
            actions.append(a)
            log_probs.append(torch.log(probs[0, a] + 1e-8))
        return actions, sum(log_probs)

# REINFORCE Trainer with entropy annealing
class REINFORCETrainer:
    def __init__(self, controller, device, lr=3e-3, ema_decay=0.95,
                 entropy_weight_start=0.02, entropy_weight_end=0.005,
                 curiosity_bonus=0.02, total_episodes=200, proxy_epochs=10):
        self.controller = controller.to(device)
        self.device = device
        self.optimizer = optim.Adam(controller.parameters(), lr=lr)
        self.ema_decay = ema_decay
        self.entropy_weight_start = entropy_weight_start
        self.entropy_weight_end = entropy_weight_end
        self.total_episodes = total_episodes
        self.curiosity_bonus = curiosity_bonus
        self.proxy_epochs = proxy_epochs
        self.baseline = 0.0
        self.seen_archs = set()
        self.top_archs = []
        self.episode = 0

    def get_entropy_weight(self):
        progress = min(self.episode / self.total_episodes, 1.0)
        weight = self.entropy_weight_start + (self.entropy_weight_end - self.entropy_weight_start) * progress
        return max(weight, self.entropy_weight_end)

    def get_reward(self, arch_ops):
        key = tuple(arch_ops)
        bonus = self.curiosity_bonus if key not in self.seen_archs else 0.0
        self.seen_archs.add(key)
        acc = evaluate_architecture(arch_ops, self.device, n_epochs=self.proxy_epochs)
        reward = acc + bonus
        self.top_archs.append((reward, arch_ops))
        self.top_archs = sorted(self.top_archs, key=lambda x: -x[0])[:10]
        return reward

    def update(self, n_samples=5):
        self.episode += 1
        rewards = []
        log_probs = []
        for _ in range(n_samples):
            arch, lp = self.controller.sample_architecture(self.device)
            r = self.get_reward(arch)
            rewards.append(r)
            log_probs.append(lp)

        mean_r = np.mean(rewards)
        self.baseline = self.ema_decay * self.baseline + (1 - self.ema_decay) * mean_r
        advantages = [r - self.baseline for r in rewards]

        policy_loss = 0.0
        for lp, adv in zip(log_probs, advantages):
            policy_loss += -lp * adv
        policy_loss = policy_loss / n_samples

        dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        hidden = (torch.zeros(2, 1, self.controller.lstm.hidden_size, device=self.device),
                  torch.zeros(2, 1, self.controller.lstm.hidden_size, device=self.device))
        logits, _ = self.controller.forward(dummy, hidden)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum().mean()

        entropy_weight = self.get_entropy_weight()
        loss = policy_loss - entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=5.0)
        self.optimizer.step()

        return {
            'mean_reward': mean_r,
            'max_reward': max(rewards),
            'entropy': entropy.item(),
            'entropy_weight': entropy_weight,
            'n_unique_archs': len(self.seen_archs)
        }

# Main training loop
def run_search(n_episodes=200, archs_per_episode=5, proxy_epochs=10, device='cuda'):
    controller = Controller(n_ops=len(OPS_NAMES), n_cells=10)
    trainer = REINFORCETrainer(controller,
                               device=torch.device(device),
                               lr=3e-3,
                               ema_decay=0.95,
                               entropy_weight_start=0.02,
                               entropy_weight_end=0.005,
                               curiosity_bonus=0.02,
                               total_episodes=n_episodes,
                               proxy_epochs=proxy_epochs)
    wandb.init(project='rl-nas', name='reinforce-retrain',
               config={'episodes': n_episodes,
                       'archs_per_episode': archs_per_episode,
                       'proxy_epochs': proxy_epochs,
                       'lr': 3e-3,
                       'entropy_start': 0.02,
                       'entropy_end': 0.005})
    best_mean = -float('inf')
    for ep in range(1, n_episodes + 1):
        res = trainer.update(n_samples=archs_per_episode)
        wandb.log(res, step=ep)
        if res['mean_reward'] > best_mean:
            best_mean = res['mean_reward']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(controller.state_dict(), 'checkpoints/controller_best.pth')
        if ep % 10 == 0:
            print(f"Episode {ep}: mean reward={res['mean_reward']:.2f}, "
                  f"entropy={res['entropy']:.4f}, weight={res['entropy_weight']:.4f}")

    with open('checkpoints/top_archs.json', 'w') as f:
        json.dump([{'reward': r, 'arch': a} for r, a in trainer.top_archs], f, indent=2)
    wandb.finish()
    print(f"Best mean reward: {best_mean:.2f}")
    print("Top architectures saved to checkpoints/top_archs.json")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_search(n_episodes=30, archs_per_episode=5, proxy_epochs=10, device=device)
