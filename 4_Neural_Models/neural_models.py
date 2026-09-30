import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    torch.manual_seed(seed)

# Task 1 & 2 definitions
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_bin = torch.tensor([[0.], [1.], [1.], [0.]])

def train_xor(activation_fn, init_zeros=False):
    set_seed(42)
    model = nn.Sequential(
        nn.Linear(2, 2),
        activation_fn(),
        nn.Linear(2, 1)
    )
    
    if init_zeros:
        for param in model.parameters():
            nn.init.zeros_(param)
            
    criterion = nn.BCEWithLogitsLoss()
    # Using a higher learning rate and more steps to ensure convergence for XOR
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    initial_loss = None
    early_grad_norm = None
    first_layer_grad = None
    
    for step in range(10000):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y_bin)
        
        if step == 0:
            initial_loss = loss.item()
            
        loss.backward()
        
        if step == 0:
            early_grad_norm = torch.norm(model[0].weight.grad).item()
            first_layer_grad = model[0].weight.grad.clone()
            
        optimizer.step()
        
    final_loss = loss.item()
    probs = torch.sigmoid(model(X))
    preds = (probs > 0.5).float()
    all_correct = torch.all(preds == y_bin).item()
    
    return initial_loss, final_loss, probs, preds, all_correct, early_grad_norm, first_layer_grad, model

# Part A & B
print("--- Part A & B: Basic XOR with Sigmoid ---")
i_loss, f_loss, probs, preds, corr, grad_norm, first_grad, trained_model = train_xor(nn.Sigmoid)
print(f"Initial Loss: {i_loss:.4f}, Final Loss: {f_loss:.4f}")
print("Final Probabilities:\n", probs.detach())
print("Predictions:\n", preds.detach())
print(f"All Correct: {corr}")
print("First layer gradient (step 0):\n", first_grad)

# Part C: Symmetry
print("\n--- Part C: Symmetry Experiment (Init Zeros) ---")
_, f_loss_z, _, _, _, _, _, trained_model_z = train_xor(nn.Sigmoid, init_zeros=True)
print(f"Final Loss: {f_loss_z:.4f}")
print("Hidden weights after training:\n", trained_model_z[0].weight.data)

# Part D: Activation
print("\n--- Part D: Activation Experiment ---")
acts = {"Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "ReLU": nn.ReLU}
print(f"{'Activation':10} | {'Final Loss':10} | {'4/4 correct?':12} | {'Early Grad Norm'}")
for name, act in acts.items():
    _, f_loss_a, _, _, corr_a, e_grad_a, _, _ = train_xor(act)
    print(f"{name:10} | {f_loss_a:10.4f} | {str(corr_a):12} | {e_grad_a:.4f}")


# Part E: 3-class XOR
print("\n--- Part E: 3-Class XOR ---")
y_multi = torch.tensor([0, 1, 1, 2], dtype=torch.long)

set_seed(42)
model_multi = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),
    nn.Linear(2, 3)
)

criterion_multi = nn.CrossEntropyLoss()
optimizer_multi = optim.SGD(model_multi.parameters(), lr=0.5)

for step in range(10000):
    optimizer_multi.zero_grad()
    logits = model_multi(X)
    loss = criterion_multi(logits, y_multi)
    loss.backward()
    optimizer_multi.step()

final_logits = model_multi(X).detach()
probs_multi = torch.softmax(final_logits, dim=1)
preds_multi = torch.argmax(probs_multi, dim=1)

print("Final Loss:", loss.item())
print("Predicted probabilities:\n", probs_multi)
print("Predictions:", preds_multi)
sum_probs = probs_multi[0].sum().item()
print("Sum of probabilities for first example:", sum_probs)

# Optional diagnostic
print("\n--- Optional Diagnostic ---")
logits_shifted = final_logits + 100.0
probs_shifted = torch.softmax(logits_shifted, dim=1)
print("Shifted probabilities (should match original):\n", probs_shifted)
