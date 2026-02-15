# 🚢 Titanic Survival Prediction with PyTorch
### *High-Stability Deep Learning Pipeline for Binary Classification*

This repository features a robust, end-to-end deep learning pipeline designed to predict passenger survival on the Titanic. The project focuses on **Gradient Stability** and **Numerical Optimization**, leveraging PyTorch to handle traditional neural network pitfalls like vanishing gradients and overfitting.

## ✨ Key Features
*   **🛡️ Anti-Vanishing Architecture:** Implements `LeakyReLU` activations and `He/Kaiming` weight initialization to ensure consistent gradient flow across layers.
*   **⚖️ Internal Normalization:** Utilizes `BatchNorm1d` layers to reduce internal covariate shift and accelerate convergence.
*   **🎯 Numerical Stability:** Employs `BCEWithLogitsLoss` for a more stable computation of the loss function compared to manual Sigmoid + BCELoss.
*   **📉 Dynamic Learning:** Features a `StepLR` scheduler and `Adam` optimizer with weight decay to fine-tune the model during later stages of training.
*   **🛡️ Gradient Clipping:** Prevents exploding gradients by rescaling gradients during the backpropagation process.

## 🛠 Tech Stack
*   **Framework:** PyTorch
*   **Data Analysis:** Pandas, Seaborn
*   **Preprocessing:** Scikit-Learn (StandardScaler, One-Hot Encoding)
*   **Visualization:** Matplotlib

## 🔍 Feature Engineering Details
The model transforms the raw Titanic dataset into high-signal variables through a custom engineering pipeline:

| Feature | Description |
| :--- | :--- |
| **family_size** | Combined count of siblings, spouses, parents, and children. |
| **is_alone** | Binary indicator for passengers traveling without family. |
| **title** | Extracted status from the 'who' column (man, woman, child). |
| **Group-based Imputation** | Missing ages filled using the median of specific `pclass` and `sex` groups. |

## 📊 Model Performance
The architecture is designed to maximize test accuracy while maintaining a smooth loss curve. 

*   **Optimization:** The model runs for 1,000 epochs with automated learning rate reduction.
*   **Monitoring:** Real-time tracking of training loss and test accuracy to identify the peak performance epoch.
