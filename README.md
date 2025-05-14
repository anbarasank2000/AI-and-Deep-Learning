# ENPM703: Image Classification


This repository is a collection of image classification projects completed for **ENPM703: AI and Deep Learning** at the University of Maryland. It includes a range of machine learning and deep learning approaches applied to the **CIFAR-10** dataset, starting from classical methods like KNN and SVM to deep neural networks implemented manually and using PyTorch.

---

## 📂 Repository Structure

```
.
├── assignment1
│   ├── knn_classifier.py
│   ├── svm_classifier.py
│   ├── softmax_classifier.py
│   ├── two_layer_nn.py
│   └── feature_extraction.py
├── assignment2
│   ├── fully_connected_net.py
│   ├── batchnorm.py
│   ├── cnn_layers.py
│   ├── dropout.py
│   └── cnn_pytorch.py
├── assignment3
│   ├── rnn_captioning.py
│   └── word_embedding.py
└── README.md
```

---

## 📚 Project Breakdown

### Assignment 1: Classical ML & Neural Nets

#### 1. K-Nearest Neighbors (KNN)
- Vectorized L2 distance computation (zero, one, two loops).
- Cross-validation for optimal `k`.
- Accuracy: ~28.7%

#### 2. Support Vector Machine (SVM)
- Vectorized loss and gradient computation.
- Hyperparameter tuning: learning rate & regularization.
- Accuracy: ~35-40%

#### 3. Softmax Classifier
- Converts class scores into probabilities.
- Cross-entropy loss optimization.
- Smooth gradient transitions.
- Accuracy: ~40-45%

#### 4. Two-Layer Neural Network
- ReLU activation, Softmax loss.
- Backpropagation using chain rule.
- Tuning: `lr`, `reg`, hidden layer size.
- Accuracy: ~50-55%

#### 5. Image Feature Extraction
- **HOG** for edge/texture patterns.
- **HSV Histogram** for color features.
- Enhanced performance with both SVM & NN.

---

### Assignment 2: Deep Learning Foundations

#### 1. Fully Connected Network
- Stack of affine → ReLU → softmax.
- Optimizers: SGD+Momentum, RMSProp, Adam.
- Hyperparameter search on `lr` and `weight_scale`.

#### 2. Batch Normalization
- Implemented vanilla and simplified batchnorm.
- Compared with LayerNorm.
- Observed improved convergence and stability.

#### 3. Convolutional Neural Network (CNN)
- Manual implementation of Conv, ReLU, MaxPool layers.
- Compared nested loop vs optimized convolution.
- CNN properties:
  - Local receptive fields
  - Hierarchical feature learning
  - Spatial invariance via pooling

#### 4. Dropout
- Vanilla vs Inverted dropout.
- Reduced overfitting and improved generalization.
- Trade-off: lower training accuracy, better validation accuracy.

#### 5. PyTorch CNN
- Architecture:
  - 2 Conv Layers: (5x5, 64ch), (3x3, 32ch)
  - BatchNorm, ReLU, MaxPool
  - FC layers: 256 → 128 → 10
  - Dropout & Adam Optimizer
- Best config: `lr=1e-3`, `weight_decay=1e-4`
- Accuracy: ~75-80%

---

### Assignment 3: Sequence Modeling

#### Recurrent Neural Network (RNN)
- Vanilla RNN implementation:
  ```
  h_t = tanh(Wxh * x_t + Whh * h_{t-1} + b)
  ```
- Word Embeddings + Temporal Affine + Softmax Loss.
- Application: Image Captioning
  - Image features → Initial hidden state
  - Captions (words) → word embeddings
  - RNN outputs vocabulary scores at each time step
  - Loss computed using temporal softmax

---

## 🌎 Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html):
- 60,000 images, 10 classes, 32x32 RGB
- Split: 50,000 train / 10,000 test
- Subsampled for experiments (e.g., 5000 train, 500 test)

---

## 📁 Requirements

- Python 3.8+
- NumPy
- Matplotlib
- PyTorch
- scikit-learn

---


## 💡 Acknowledgements

- Prof. George Zaki for course content and guidance.
- CIFAR-10 dataset by Alex Krizhevsky.
