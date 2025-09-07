# 🌀 Non-linear Classification with PyTorch

This project demonstrates why non-linear activations are crucial in neural networks using the classic `make_circles` dataset.

## 🔑 Key Points

- **Decision Boundary**: The border where the model switches between classes.  
- **Model V1**: Only linear layers → fails (can only learn straight lines).  
- **Model V2**: Linear + ReLU → learns curved boundaries, fits circles.  
- **Loss**: `BCEWithLogitsLoss` for stable binary classification.

## 📊 Result

- Linear-only model underfits.  
- Non-linear model successfully separates concentric circles.

```bash
pip install torch torchvision torchaudio
```
## Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
