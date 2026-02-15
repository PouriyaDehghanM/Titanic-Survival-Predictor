🚢 Titanic Survival Predictor (PyTorch)
Advanced Binary Classification with Gradient Stability Optimization
This project implements a deep neural network using PyTorch to predict passenger survival on the Titanic. The implementation focuses on overcoming common deep learning hurdles like Vanishing and Exploding Gradients through modern architectural choices.
🚀 Key Technical Features
To ensure stable training and high convergence speed, the model incorporates several "Robust-Net" features:
Anti-Vanishing Gradient: Uses LeakyReLU activation and He/Kaiming initialization to keep neurons active and gradients flowing.
Internal Covariate Shift Reduction: Integrated Batch Normalization layers after each linear transformation.
Numerical Stability: Utilizes BCEWithLogitsLoss which combines a Sigmoid layer and BCELoss in one single class for better numerical stability.
Regularization: Implemented L2 Weight Decay and Gradient Clipping to prevent overfitting and exploding gradients.
Learning Rate Scheduling: Uses a StepLR scheduler to fine-tune the learning process as it approaches local minima.
📊 Data Engineering
The preprocessing pipeline includes:
Feature Engineering: Creation of family_size, is_alone, and title extraction from the who column.
Imputation: Smart age filling based on the median of pclass and sex groups.
Scaling: Standard scaling applied to continuous features to ensure uniform gradient updates.
