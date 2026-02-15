🚢 Titanic Survival Predictor: PyTorch Implementation
This repository contains a Deep Learning model created with PyTorch to predict passenger survival on the Titanic. The project emphasizes stability and uses specific techniques to address common training problems in neural networks, such as Vanishing Gradients and Overfitting.
🚀 Key Features
Anti-Vanishing Gradient Architecture: Uses LeakyReLU activation functions and He/Kaiming weight initialization.
Numerical Stability: Uses BCEWithLogitsLoss for better numerical precision during backpropagation.
Batch Normalization: Applied after linear layers to speed up training and provide regularization.
Smart Feature Engineering: Includes custom features like family_size, is_alone, and title extraction from the Titanic dataset.
Gradient Management: Uses Gradient Clipping to stop exploding gradients and a Learning Rate Scheduler for smooth convergence.
🛠️ Tech Stack
Framework: PyTorch
Data Handling: Pandas, NumPy, Scikit-learn
Visualization: Matplotlib, Seaborn
🧠 Model Architecture
The network (TitanicRobustNet) has:
Input Layer: Matching the engineered feature dimensions.
Hidden Layer 1: 32 Neurons, Batch Norm, LeakyReLU.
Hidden Layer 2: 16 Neurons, Batch Norm, LeakyReLU.
Output Layer: 1 Neuron (Logits).
📊 Performance & Results
The model is trained over 1,000 epochs with an Adam optimizer.
Optimization: L2 Regularization (Weight Decay) is used to prevent overfitting.
Max Accuracy: The model regularly achieves high accuracy, typically around 80-83% (depending on the random seed) on the test set.
Metric	Status
Loss Function	Binary Cross Entropy with Logits
Optimizer	Adam (LR=0.01)
Regularization	Weight Decay, Gradient Clipping
