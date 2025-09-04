import pandas as pd
import sklearn
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np

# 1) Create a non-linearly separable binary classification toy dataset.
#    make_circles generates two concentric rings (class 0 = inner ring, class 1 = outer ring).
#    This is a classic example where a linear classifier fails, pushing us to use non-linear models.
X, y = make_circles(
    n_samples=1000,
    noise=0.03,         # add slight jitter so points are not perfectly on a circle
    random_state=42,
)

# 2) Convert to PyTorch tensors.
#    BCEWithLogitsLoss expects float targets shaped like the model output; we'll keep y as float.
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# 3) Train/test split. The split is random but reproducible thanks to random_state.
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,
    random_state=42
)

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plot the model's decision boundary over the 2D feature space.

    WHAT IS A DECISION BOUNDARY?
    ----------------------------
    For a classifier, the decision boundary is the set of points where the model is
    indifferent between classes (e.g., for binary classification, where predicted probability = 0.5).
    Visually, it's the "border" that separates the input space into regions assigned to different classes.

    HOW THIS FUNCTION WORKS
    -----------------------
    1) Move model and data to CPU (NumPy + Matplotlib interop is smoother on CPU).
    2) Build a dense grid covering the min/max of X's two features.
    3) For every grid point, run a forward pass to get logits.
    4) Convert logits to class labels:
       - Binary case: apply sigmoid -> threshold at 0.5 (we use round) to get {0,1}.
       - Multi-class case: apply softmax -> argmax to get the most likely class index.
    5) Reshape predictions to the grid shape and draw a filled contour (background colors).
    6) Overlay the real data points for reference.

    NOTES
    -----
    - We use torch.inference_mode() to avoid gradient tracking during visualization.
    - We pass raw logits to the loss function during training for numerical stability (BCEWithLogitsLoss),
      but for visualization/metrics we convert logits -> probabilities -> labels.
    """
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Build a mesh grid over the feature space
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 101),
        np.linspace(y_min, y_max, 101)
    )

    # Prepare grid points for prediction: shape (num_points, 2)
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Binary vs multi-class handling
    if len(torch.unique(y)) > 2:
        # Multi-class: logits -> probabilities -> class indices
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        # Binary: logits -> probabilities via sigmoid -> threshold at 0.5
        y_pred = torch.round(torch.sigmoid(y_logits))

    # Reshape to grid and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

class CircleModelV1(nn.Module):
    """
    A stack of linear layers WITHOUT non-linear activations.
    IMPORTANT: A composition of linear (affine) layers without non-linearities
    collapses to a single linear transform. That means this model can only learn
    linear decision boundaries (a straight line in 2D). It will struggle on make_circles.
    """
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x):
        # Equivalent to a single affine map overall (no non-linearities in-between).
        return self.layer_3(self.layer_2(self.layer_1(x)))

class CircleModelV2(nn.Module):
    """
    Same structure but WITH non-linear activations (ReLU) between linear layers.
    Non-linear activations are crucial: they allow the network to learn curved,
    complex decision boundaries (exactly what's needed for concentric circles).
    """
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # element-wise non-linearity

    def forward(self, x):
        # Linear -> ReLU -> Linear -> ReLU -> Linear (logit)
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

def accuracy_fn(y_true, y_pred):
    """
    Simple accuracy for binary labels in {0,1}.
    y_pred should already be thresholded (i.e., labels, not probabilities).
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model 1: no non-linearities (expected to underfit make_circles)
model_1 = CircleModelV1().to(device)

# BCEWithLogitsLoss = Binary Cross-Entropy with logits.
# It combines a sigmoid layer + binary cross-entropy in a numerically stable way.
# Pass RAW LOGITS to it (do NOT apply sigmoid before the loss).
loss_fn = nn.BCEWithLogitsLoss()

# Basic optimizer; a higher lr speeds learning on this simple toy but may oscillate on harder tasks.
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)

torch.manual_seed(42)

epochs = 1000

# Move data to device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test   = X_test.to(device),  y_test.to(device)

for epoch in range(epochs):
    model_1.train()

    # Forward pass: outputs are logits (unnormalized scores)
    y_logits = model_1(X_train).squeeze()

    # For accuracy, convert logits -> probabilities -> labels
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Compute loss on logits vs. float targets
    loss = loss_fn(y_logits, y_train)

    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation on test set (no grad, no dropout/bn updates)
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"[Model_1] Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | "
              f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# Model 3: with non-linearities (ReLU) — should learn a curved boundary that fits circles
model_3 = CircleModelV2().to(device)
print(model_3)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

torch.manual_seed(42)
epochs = 10000  # longer training to let the non-linear model fully carve the boundary

# Ensure data is on device (already is, but kept explicit/defensive)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test   = X_test.to(device),  y_test.to(device)

for epoch in range(epochs):
    model_3.train()

    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 1000 == 0:
        print(f"[Model_3] Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | "
              f"Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# Make final predictions on the test set with the non-linear model
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

print("Sample preds vs labels:")
print(y_preds[:10].cpu())
print(y_test[:10].cpu())

# Compare decision boundaries of the linear vs non-linear models
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Decision boundary — Linear-only model (expected to fail on circles)")
plot_decision_boundary(model_1, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Decision boundary — Non-linear model (ReLU) fits circles")
plot_decision_boundary(model_3, X_test, y_test)

plt.tight_layout()
plt.show()
