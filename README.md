# 🧠 Neural Network from Scratch (NumPy)

## 📌 Overview

This project implements a simple feedforward neural network **from scratch using only NumPy**, without any deep learning frameworks.
The goal is to demonstrate a clear understanding of:

* Forward propagation
* Backpropagation
* Gradient-based optimization
* Multi-class classification

The model is trained on the classic **Iris dataset**.

---

## 📊 Dataset

We use the Iris dataset:

* 150 samples
* 4 features
* 3 classes

Before training:

* Features are **standardized**
* Labels are converted to **one-hot encoding**

---

## 🏗️ Model Architecture

A simple 2-layer neural network:

$$
X \rightarrow \text{Linear} \rightarrow \text{ReLU} \rightarrow \text{Linear} \rightarrow \text{Softmax}
$$

### Dimensions:

* Input: $x \in \mathbb{R}^4$
* Hidden layer: $h \in \mathbb{R}^{10}$
* Output: $y \in \mathbb{R}^{3}$

---

## ⚙️ Forward Pass

### 1. First Layer

$$
A^{(1)} = XW_1 + b_1
$$

$$
H^{(1)} = \text{ReLU}(A^{(1)})
$$

ReLU activation:
$$
\text{ReLU}(x) = \max(0, x)
$$

---

### 2. Second Layer

$$
A^{(2)} = H^{(1)}W_2 + b_2
$$

---

### 3. Softmax Output

$$
Z_i = \frac{e^{A_i}}{\sum_j e^{A_j}}
$$

This converts logits into probabilities.

---

## 📉 Loss Function

We use **Cross-Entropy Loss**:

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{C} y_{ik} \log(z_{ik})
$$

Where:

* $y$ — true labels (one-hot)
* $z$ — predicted probabilities
* $m$ — batch size

---

## 🔁 Backpropagation

### Output Layer Gradient

$$
\frac{\partial \mathcal{L}}{\partial A^{(2)}} = Z - Y
$$

---

### Gradients for second layer

$$
\frac{\partial \mathcal{L}}{\partial W_2} = \frac{H^T (Z - Y)}{m}
$$

$$
\frac{\partial \mathcal{L}}{\partial b_2} = \frac{1}{m} \sum (Z - Y)
$$

---

### Hidden Layer Gradient

$$
\frac{\partial \mathcal{L}}{\partial H^{(1)}} = (Z - Y) W_2^T
$$

$$
\frac{\partial \mathcal{L}}{\partial A^{(1)}} = \frac{\partial \mathcal{L}}{\partial H^{(1)}} \cdot \mathbb{1}(A^{(1)} > 0)
$$

---

### Gradients for first layer

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{X^T \frac{\partial \mathcal{L}}{\partial A^{(1)}}}{m}
$$

$$
\frac{\partial \mathcal{L}}{\partial b_1} = \frac{1}{m} \sum \frac{\partial \mathcal{L}}{\partial A^{(1)}}
$$

---

## 🔧 Optimization

We use simple **Gradient Descent**:

$$
W := W - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W}
$$

$$
b := b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}
$$

Where:

* $\alpha$ — learning rate

---

## ⚡ Initialization

Weights are initialized using **He Initialization**:

$$
W \sim \mathcal{N}(0, \sqrt{\frac{2}{n}})
$$

This helps prevent vanishing/exploding gradients.

---

## 🔄 Training Procedure

1. Shuffle dataset each epoch
2. Split into mini-batches
3. Perform:

   * Forward pass
   * Loss computation
   * Backward pass
   * Parameter update

---

## 📈 Evaluation

Accuracy is computed as:

$$
\text{Accuracy} = \frac{1}{N} \sum \mathbb{1}(\hat{y} = y)
$$

---

## 🚀 Results

The model typically achieves:

* High training accuracy
* Strong generalization on test data

Despite its simplicity, this implementation demonstrates the full training pipeline of a neural network.

---

## 🎯 Key Takeaways

* Neural networks can be implemented with **pure NumPy**
* Backpropagation is just **chain rule + matrix operations**
* Proper initialization and normalization are critical
* Even simple models can perform well on structured data

---

## 📚 Future Improvements

* Add regularization (L2, Dropout)
* Try deeper architectures
* Implement different optimizers (Adam, RMSprop)
* Add learning rate scheduling

---

## 🧑‍💻 Author

Implemented from scratch for educational purposes to deeply understand how neural networks work under the hood.
