import torch
import torch.nn.functional as F
from src.search_space import Cell
from src.supernet import Supernet
import torchvision
import torchvision.transforms as transforms

device = 'cuda'

# ── CHECK 1: Does a single forward pass produce non-random output? ──
print("=== CHECK 1: Forward pass ===")
model = Supernet(C=16, n_cells=10, n_classes=10).to(device) # type: ignore
x = torch.randn(4, 3, 32, 32).to(device)
with torch.no_grad():
    out = model(x)
probs = F.softmax(out, dim=1)
print(f"Output shape: {out.shape}")           # must be (4, 10)
print(f"Max prob: {probs.max().item():.3f}")  # if ~0.1 = uniform = bad init is ok
print(f"Any NaN: {torch.isnan(out).any()}")   # must be False
print(f"Any Inf: {torch.isinf(out).any()}")   # must be False

# ── CHECK 2: Does one training step change the weights? ──
print("\n=== CHECK 2: Gradient flow ===")
model.train()
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if not p.requires_grad == False],
    lr=0.025, momentum=0.9, weight_decay=3e-4
)
x = torch.randn(4, 3, 32, 32).to(device)
y = torch.randint(0, 10, (4,)).to(device)

# save weights before
first_conv = list(model.parameters())[0].clone().detach()

optimizer.zero_grad()
out = model(x)
loss = F.cross_entropy(out, y)
loss.backward()

# check gradients exist
grad_norms = []
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norms.append(p.grad.norm().item())
print(f"Params with gradients: {len(grad_norms)}")  # must be > 0
print(f"Mean grad norm: {sum(grad_norms)/len(grad_norms):.6f}" if grad_norms else "NO GRADIENTS — BUG!")

optimizer.step()
first_conv_after = list(model.parameters())[0].clone().detach()
changed = not torch.allclose(first_conv, first_conv_after)
print(f"Weights changed after step: {changed}")  # must be True

# ── CHECK 3: DataLoader sanity ──
print("\n=== CHECK 3: DataLoader ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
imgs, labels = next(iter(loader))
print(f"Image shape: {imgs.shape}")           # must be (4,3,32,32)
print(f"Image range: [{imgs.min():.2f}, {imgs.max():.2f}]")  # must be ~[-2, 2]
print(f"Labels: {labels}")                     # must be ints 0-9
print("\nDone. Paste all output back to Claude.")
