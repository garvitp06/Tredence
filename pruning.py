import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# --- Part 1: The "Prunable" Linear Layer ---
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Gate scores: initialized to a high value so gates start near 1 (open)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        # Initialize gate_scores to 0.5 so they start at sigmoid(0.5)=0.62
        # This allows L1 penalty to push them below -4.6 (1e-2) within 10 epochs
        nn.init.constant_(self.gate_scores, 0.5)

    def forward(self, x):
        # Transform scores to [0, 1] gates
        gates = torch.sigmoid(self.gate_scores)
        # Element-wise weight pruning
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity_loss(self):
        # L1 norm of the sigmoid gates
        return torch.sum(torch.sigmoid(self.gate_scores))


# --- Part 2: Network Definition ---
class PrunableNet(nn.Module):
    def __init__(self):
        super(PrunableNet, self).__init__()
        # MNIST images are 28x28 grayscale (784 pixels)
        self.fc1 = PrunableLinear(28 * 28, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def total_sparsity_loss(self):
        return self.fc1.get_sparsity_loss() + \
            self.fc2.get_sparsity_loss() + \
            self.fc3.get_sparsity_loss()

    def get_sparsity_stats(self, threshold=1e-2):
        total_weights = 0
        pruned_weights = 0
        all_gates = []

        for layer in [self.fc1, self.fc2, self.fc3]:
            gates = torch.sigmoid(layer.gate_scores).detach()
            all_gates.append(gates.view(-1))
            total_weights += gates.numel()
            pruned_weights += torch.sum(gates < threshold).item()

        sparsity_pct = (pruned_weights / total_weights) * 100
        return sparsity_pct, torch.cat(all_gates).cpu().numpy()


# --- Part 3: Training Loop ---
def train_and_evaluate(lambd, epochs=10):
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    model = PrunableNet().to(device)
    # Using a slightly higher learning rate so gates can travel to -4.6 threshold
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)
            # Sparsity Loss is normalized by total weights to keep lambda scale consistent
            sparsity_loss = model.total_sparsity_loss()

            total_loss = ce_loss + lambd * sparsity_loss
            total_loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    sparsity, gates_dist = model.get_sparsity_stats()
    return accuracy, sparsity, gates_dist


# Run Experiments
if __name__ == '__main__':
    lambdas = [1e-5, 1e-4, 1e-3]
    results = {}

    for l in lambdas:
        print(f"Running Experiment for Lambda: {l}")
        # Running for 10 epochs allows proper gate pruning dynamics to settle
        acc, sp, dist = train_and_evaluate(l, epochs=10) 
        print(f"Lambda: {l}, Accuracy: {acc:.2f}%, Sparsity: {sp:.2f}%")
        results[l] = (acc, sp, dist)

    # Plotting the best model gate distribution (using high lambda to show spike)
    best_l = 1e-3
    _, _, gate_values = results[best_l]
    plt.hist(gate_values, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Gate Value Distribution (Lambda={best_l})')
    plt.xlabel('Gate Value (Sigmoid Output)')
    plt.ylabel('Frequency')
    plt.savefig('gate_distribution.png')
    print("Saved gate_distribution.png")