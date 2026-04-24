# 🧠 Self-Pruning Neural Network

This project implements a neural network that **learns to prune itself during training** using a differentiable gating mechanism and sparsity regularization.

Instead of removing weights after training, the model dynamically identifies and suppresses less important connections during learning.

---

## 🚀 Key Idea

Each weight is associated with a learnable gate:

    pruned_weight = weight × sigmoid(gate_score)

- Gate ≈ 1 → connection is active  
- Gate ≈ 0 → connection is effectively pruned  

This allows the model to optimize both its **weights and structure simultaneously**.

---

## ⚙️ Methodology

### 🔹 Prunable Linear Layer
- Custom linear layer with learnable gate parameters
- Gates are applied to weights using a sigmoid function

### 🔹 Sparsity Regularization
- L1 penalty applied on gate values:
  
      Loss = CrossEntropy + λ × SparsityLoss

- Encourages many gates to become zero → sparse network

### 🔹 Progressive Pruning
- Sparsity strength increases over time:

      λ_current = λ × (epoch / total_epochs)

- Early training → focus on accuracy  
- Later training → focus on pruning  

### 🔹 Hard Pruning
- After training, gates below a threshold are permanently removed
- Converts soft sparsity into real compression

---

## 📊 Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 1e-5   | 55.60%   | 8.51%    |
| 1e-4   | 56.60%   | 40.85%   |
| 1e-3   | 51.05%   | 73.41%   |

👉 **Best trade-off achieved at λ = 1e-4**, where the model improves accuracy while pruning ~40% of weights.

---

## 📈 Visualizations

### 🔹 Sparsity vs Epochs
- Shows gradual pruning during training

### 🔹 Gate Distribution
- Strong peak near 0 → many weights pruned
- Some values near 1 → important connections retained

---

## 🧠 Key Insights

- Neural networks can learn to optimize their own structure during training  
- L1 regularization effectively drives sparsity  
- Moderate pruning improves generalization  
- There is a clear trade-off between model efficiency and performance  

---

## 🛠️ How to Run

```bash
pip install torch torchvision matplotlib
python train_prunable.py
