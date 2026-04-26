# Neural Network from Scratch (NumPy)

This project implements a simple feedforward neural network **from scratch using only NumPy**, without any deep learning frameworks such as TensorFlow or PyTorch.

The main goal of this project is to understand what happens under the hood in modern deep learning frameworks. Instead of treating neural networks as a “black box”, this project focuses on the core mathematical components behind them.

The implementation is explained in three main parts:

- Forward propagation  
- Backpropagation  
- Gradient-based optimization  

## Forward propagation

The dataset contains 150 samples, each with 4 input features and belonging to one of 3 classes. This is a **multi-class classification problem**.

### Model Architecture

Based on the dataset:

- Input layer: 4 neurons (one per feature)  
- Hidden layer: 10 neurons  
- Output layer: 3 neurons (one per class)  

---

### Weight Initialization

Before performing forward propagation, we need to initialize the model parameters (weights and biases). 

These parameters are the “brain” of the model — they determine how input data is transformed into predictions.

Weights are initialized randomly, but not purely random.  
We use **He initialization**, which helps stabilize training and improves convergence during gradient-based optimization.

```python
self.W1 = np.random.randn(input_dim, h_dim) * np.sqrt(2 / input_dim)
self.B1 = np.zeros((1, h_dim))

self.W2 = np.random.randn(h_dim, out_dim) * np.sqrt(2 / h_dim)
self.B2 = np.zeros((1, out_dim))
````
To better understand the implementation, let's rewrite the model in mathematical form using matrices.

Neural networks rely heavily on matrix operations, which allow us to compute many operations in parallel.  
This is one of the key reasons why neural networks are efficient and scalable.

For now, let's define the shapes of our parameters:

- W₁ ∈ ℝ^(4 × 10)  
- B₁ ∈ ℝ^(1 × 10)  
- W₂ ∈ ℝ^(10 × 3)  
- B₂ ∈ ℝ^(1 × 3)  

Below is how these parameters look in matrix form:

$$
W_1 =
\begin{bmatrix}
w^{(1)}_{11} & w^{(1)}_{12} & w^{(1)}_{13} & \cdots & w^{(1)}_{1,10} \\
w^{(1)}_{21} & w^{(1)}_{22} & w^{(1)}_{23} & \cdots & w^{(1)}_{2,10} \\
w^{(1)}_{31} & w^{(1)}_{32} & w^{(1)}_{33} & \cdots & w^{(1)}_{3,10} \\
w^{(1)}_{41} & w^{(1)}_{42} & w^{(1)}_{43} & \cdots & w^{(1)}_{4,10}
\end{bmatrix}
\quad
B_1 =
\begin{bmatrix}
b^{(1)}_1 & b^{(1)}_2 & b^{(1)}_3 & \cdots & b^{(1)}_{10}
\end{bmatrix}
$$

$$
W_2 =
\begin{bmatrix}
w^{(2)}_{11} & w^{(2)}_{12} & w^{(2)}_{13} \\
w^{(2)}_{21} & w^{(2)}_{22} & w^{(2)}_{23} \\
\vdots & \vdots & \vdots \\
w^{(2)}_{10,1} & w^{(2)}_{10,2} & w^{(2)}_{10,3}
\end{bmatrix}
\quad
B_2 =
\begin{bmatrix}
b^{(2)}_1 & b^{(2)}_2 & b^{(2)}_3
\end{bmatrix}
$$

### ReLU Activation

```python
def relu(self, x):
        return np.maximum(0, x)
```

ReLU sets negative values to zero and introduces non-linearity.  
Without activation functions, a neural network would behave like a linear model.  
This allows the model to learn complex patterns and helps reduce the vanishing gradient problem.


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
