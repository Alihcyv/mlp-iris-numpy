# Neural Network from Scratch (NumPy)

This project implements a simple feedforward neural network **from scratch using only NumPy**, without any deep learning frameworks such as TensorFlow or PyTorch.

The main goal of this project is to understand what happens under the hood in modern deep learning frameworks. Instead of treating neural networks as a “black box”, this project focuses on the core mathematical components behind them.

The implementation is explained in three main parts:

- Forward propagation  
- Backpropagation  
- Gradient-based optimization  

## Forward Propagation

Before diving into the full implementation of the forward pass, we need to cover the essential building blocks. This section explains the data preprocessing and the activation functions that make the network work.

### Data Preparation

```python
def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # normalization
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # one-hot encoding
    Y = np.eye(3)[y]

    # shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X, Y = X[indices], Y[indices]

    # split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    return X_train, X_test, Y_train, Y_test
```

To implement forward propagation, the data must first be preprocessed. The following steps are performed in the load_data function:

- [Normalization](https://www.datacamp.com/tutorial/normalization-in-machine-learning): The input features are standardized to have a mean of 0 and a standard deviation of 1. This ensures a smoother loss surface, which allows the model to converge faster and more efficiently.
- [One-Hot Encoding](https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/): Target labels are converted from scalars to one-hot encoded vectors. This representation is essential for calculating the loss and gradients during the backpropagation phase.
- Shuffling: Since the Iris dataset is sorted by class, the data is shuffled to prevent class imbalance in the training and testing sets.
- Data Splitting: The dataset is split into a training set (80% / 120 samples) and a testing set (20% / 30 samples) to evaluate the model's generalization performance.
  
---

### Model Architecture

Based on the dataset:

- Input layer: 4 neurons (one per feature)  
- Hidden layer: 10 neurons  
- Output layer: 3 neurons (one per class)

<p align="center">
  <img src="images/mlp_model_from_scratch.jpg" width="400"/>
</p>

---

### Weight Initialization

Before performing forward propagation, we need to initialize the model parameters (weights and biases). 

These parameters are the “brain” of the model — they determine how input data is transformed into predictions.

Weights are initialized randomly, but not purely random.  
We use [**He initialization**](https://medium.com/@piyushkashyap045/mastering-weight-initialization-in-neural-networks-a-beginners-guide-6066403140e9), which helps stabilize training and improves convergence during gradient-based optimization.

```python
self.W1 = np.random.randn(input_dim, h_dim) * np.sqrt(2 / input_dim)
self.B1 = np.zeros((1, h_dim))

self.W2 = np.random.randn(h_dim, out_dim) * np.sqrt(2 / h_dim)
self.B2 = np.zeros((1, out_dim))
```
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

---

### ReLU Activation Function

```python
def relu(self, x):
        return np.maximum(0, x)
```
<p align="center">
  <img src="images/relu_activation.jpg" width="500"/>
</p>

ReLU sets negative values to zero and introduces non-linearity.  
Without activation functions, a neural network would behave like a linear model.  
This allows the model to learn complex patterns and helps reduce the vanishing gradient problem.

---

### Softmax Activation Function

<p align="center">
  <img src="images/softmax.png" width="500"/>
</p>

The final layer of our network produces raw scores called **logits**. However, these scores can be any real number, making them difficult to use for classification. To solve this, we use the **Softmax** activation function.

Softmax transforms these logits into a probability distribution where:
1. Each value is between **0 and 1**.
2. The sum of all output values is exactly **1**.

This allows us to interpret the output as the model's confidence in each class and is a requirement for calculating the **Cross-Entropy Loss**. The standard mathematical formula is:

$$z_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

In practice, calculating $e^{z_i}$ can be dangerous. If a logit $z_i$ is a large number (e.g., 500), $e^{500}$ will result in an **overflow** (it becomes `inf` in Python), which crashes the model.

To prevent this, we use a technique called **Numerical Stability**. We subtract the maximum value of the input vector from all elements before computing the exponential:

$$z_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{C} e^{z_j - \max(\mathbf{z})}}$$

```python
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
```

Softmax is **shift-invariant**. This means that adding or subtracting a constant from all inputs does not change the final output probability. Mathematically:
$$\text{Softmax}(x) = \text{Softmax}(x - T)$$

By setting $T = \max(\mathbf{z})$, the largest value in the vector becomes $0$ ($e^0 = 1$), and all other values become negative. This ensures that the values never explode, keeping the calculations stable while preserving the original probability distribution.

---

### Forward Pass

Now that we have defined the data and activation functions, we can implement the forward propagation.

```python
def forward(self, X):
        self.A1 = X @ self.W1 + self.B1
        self.H1 = self.relu(self.A1)

        self.A2 = self.H1 @ self.W2 + self.B2
        self.Z = self.softmax(self.A2)

        return self.Z
```


To better understand the forward pass, consider a single input vector:

$$
X = [x_1, x_2, x_3, x_4]
$$

This represents one observation.

Now, consider a single neuron in the hidden layer.  
This neuron is connected to all input features through a set of weights:

$$[w_{11}^{(1)}, w_{21}^{(1)}, w_{31}^{(1)}, w_{41}^{(1)}]$$

The neuron computes a weighted sum of the inputs:

$$a^{(1)}_1 = (x_1 \cdot w_{11}^{(1)} + x_2 \cdot w_{21}^{(1)} + x_3 \cdot w_{31}^{(1)} + x_4 \cdot w_{41}^{(1)}) + b_1^{(1)}$$

Here, \( b \) is the bias term.

The same operation is applied to all neurons in the hidden layer.

The calculation of the vector $\mathbf{A_1}$ can be represented as a matrix operation:

$$\mathbf{A_1} = \begin{bmatrix} x_1 & x_2 & x_3 & x_4 \end{bmatrix} \cdot 
\begin{bmatrix}
w^{(1)}_{11} & w^{(1)}_{12} & w^{(1)}_{13} & \cdots & w^{(1)}_{1,10} \\
w^{(1)}_{21} & w^{(1)}_{22} & w^{(1)}_{23} & \cdots & w^{(1)}_{2,10} \\
w^{(1)}_{31} & w^{(1)}_{32} & w^{(1)}_{33} & \cdots & w^{(1)}_{3,10} \\
w^{(1)}_{41} & w^{(1)}_{42} & w^{(1)}_{43} & \cdots & w^{(1)}_{4,10}
\end{bmatrix} 
+ 
\begin{bmatrix}
b^{(1)}_1 & b^{(1)}_2 & b^{(1)}_3 & \cdots & b^{(1)}_{10}
\end{bmatrix}$$

**Where:**
- $\mathbf{X} = [x_1, x_2, x_3, x_4]$ is the **input vector** (1 $\times$ 4).
- $\mathbf{W_1}$ is the **weight matrix** (4 $\times$ 10), where each column represents the weights for a single neuron.
- $\mathbf{B_1}$ is the **bias vector** (1 $\times$ 10), adding a trainable offset to each neuron.

More generally, for any neuron $j$ in the hidden layer, the formula is:

$$a^{(1)}_j = \sum_{i=1}^{4} (x_i \cdot w_{ij}^{(1)}) + b_j^{(1)}$$

This mathematical operation is exactly what is implemented in the code as:
`self.A1 = X @ self.W1 + self.B1`

Then, the ReLU activation function is applied:

$$h_j = \text{ReLU}(\mathbf{a^{(1)}_j}) \quad \text{or} \quad h_j = \max(0, a^{(1)}_j)$$

This is implemented as: `self.H1 = self.relu(self.A1)`

After the hidden layer activation, the resulting vector $\mathbf{H_1}$ is passed to the output layer to produce the final predictions.

The output layer transforms the 10 hidden neurons into 3 output neurons (one for each Iris class). This is represented by the following matrix operation:

$$\mathbf{A_2} = \begin{bmatrix} h_1 & h_2 & \dots & h_{10} \end{bmatrix} \cdot 
\begin{bmatrix}
w^{(2)}_{11} & w^{(2)}_{12} & w^{(2)}_{13} \\
w^{(2)}_{21} & w^{(2)}_{22} & w^{(2)}_{23} \\
\vdots & \vdots & \vdots \\
w^{(2)}_{10,1} & w^{(2)}_{10,2} & w^{(2)}_{10,3}
\end{bmatrix} 
+ 
\begin{bmatrix}
b^{(2)}_1 & b^{(2)}_2 & b^{(2)}_3
\end{bmatrix}$$

**Where:**
- $\mathbf{H_1}$ is the **hidden layer output** (1 $\times$ 10).
- $\mathbf{W_2}$ is the **output weight matrix** (10 $\times$ 3).
- $\mathbf{B_2}$ is the **output bias vector** (1 $\times$ 3).
- $\mathbf{A_2}$ is the **logits vector** (1 $\times$ 3).

This is implemented as: `self.A2 = self.H1 @ self.W2 + self.B2`

To convert the raw scores ($\mathbf{A_2}$) into probabilities that sum up to 1, we use the **Softmax** function. For each output neuron $i$, the probability $z_i$ is calculated as:

$$z_i = \frac{e^{a^{(2)}_{i}}}{\sum_{j=1}^{3} e^{a^{(2)}_{j}}}$$

This is implemented as: `self.Z = self.softmax(self.A2)`

### Loss Function: Categorical Cross-Entropy

After obtaining the probabilities from the Softmax layer, we need to quantify how far the model's predictions are from the actual labels. For this, we use the [Categorical Cross-Entropy (CCE)](https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/) loss function.

For a single observation, the loss is calculated as:

$$L = -\sum_{i=1}^{C} y_i \cdot \log(z_i)$$

Where: 
- $C$ is the number of classes (in our case, $C$ = 3).
- $y_i$ is the ground truth (1 if the observation belongs to class $i$, otherwise 0).
- $z_i$ is the predicted probability for class $i$ produced by the Softmax function.

Since we train the model using batches rather than single observations, we calculate the average loss across the entire batch to ensure stable gradient updates:

$$J = \frac{1}{N} \sum_{n=1}^{N} L_n$$

Where $N$ is the batch size.

```python
def loss(self, Y, Z):
        return np.mean(-np.sum(Y * np.log(Z + 1e-15), axis=1))
```

---

## Backpropagation

Now we move to the most challenging yet critical part of the neural network: **Backpropagation**. This is the process where the model actually "learns" by adjusting its weights to minimize the loss.

In this section, we will manually derive the gradients of the loss function. This process highlights the fundamental difference between using a library like **NumPy** and using deep learning frameworks such as **PyTorch** or **TensorFlow**. While frameworks use a mechanism called *Automatic Differentiation* (Autograd) to compute gradients automatically, we will perform these calculations by hand using matrix calculus.

To avoid the complexity and inefficiency of calculating derivatives for each single weight sequentially, we use [**matrix calculus**](https://explained.ai/matrix-calculus/). This allows us to compute the gradients for entire layers in parallel, mirroring how professional frameworks operate under the hood.

Our goal is to find the gradients of the loss function $L$ with respect to each of our trainable parameters. These gradients tell us how much we need to change the weights to decrease the error:

$$\frac{\partial L}{\partial \mathbf{W_1}}, \quad \frac{\partial L}{\partial \mathbf{B_1}}, \quad \frac{\partial L}{\partial \mathbf{W_2}}, \quad \frac{\partial L}{\partial \mathbf{B_2}}$$


To keep the derivations clear, let's recap the forward propagation:
1. $\mathbf{A_1} = \mathbf{X}\mathbf{W_1} + \mathbf{B_1}$
2. $\mathbf{H_1} = \text{ReLU}(\mathbf{A_1})$
3. $\mathbf{A_2} = \mathbf{H_1}\mathbf{W_2} + \mathbf{B_2}$
4. $\mathbf{Z} = \text{Softmax}(\mathbf{A_2})$
5. $L = \text{Loss}(\mathbf{Y}, \mathbf{Z})$

### Step 1: Finding $\frac{\partial L}{\partial \mathbf{Z}}$

Let's follow the chain from the end of the architecture back to the weights. We start by calculating the gradient of the loss function with respect to the output of the softmax layer, $\frac{\partial L}{\partial \mathbf{Z}}$.

#### 1. Case: Batch size $N = 1$
To simplify the understanding, let's first assume that the batch size is 1. In this case, the gradient $\frac{\partial L}{\partial \mathbf{Z}}$ is a vector:

$$\frac{\partial L}{\partial \mathbf{Z}} = \left[ \frac{\partial L}{\partial z_1}, \frac{\partial L}{\partial z_2}, \frac{\partial L}{\partial z_3} \right]$$

Expanding the loss function (Cross-Entropy), we get:

$$\frac{\partial L}{\partial \mathbf{Z}} = \begin{bmatrix} 
\frac{\partial (y_1 \log(z_1) + y_2 \log(z_2) + y_3 \log(z_3))}{\partial z_1} \\ 
\frac{\partial (y_1 \log(z_1) + y_2 \log(z_2) + y_3 \log(z_3))}{\partial z_2} \\ 
\frac{\partial (y_1 \log(z_1) + y_2 \log(z_2) + y_3 \log(z_3))}{\partial z_3} 
\end{bmatrix}$$

Since each $z_i$ only depends on its corresponding $\log(z_i)$ term and the derivatives of all other terms with respect to $z_i$ are zero, we obtain:

$$\frac{\partial L}{\partial \mathbf{Z}} = \begin{bmatrix} 
-\frac{y_1}{z_1} \\ 
-\frac{y_2}{z_2} \\ 
-\frac{y_3}{z_3} 
\end{bmatrix}$$

#### 2. Case: Batch size $N > 1$
Now, we can generalize this for a batch of size $N$. The gradient $\frac{\partial L}{\partial \mathbf{Z}}$ becomes a matrix:

$$\frac{\partial L}{\partial \mathbf{Z}} = \begin{bmatrix} 
\frac{\partial L}{\partial z_{11}} & \frac{\partial L}{\partial z_{12}} & \frac{\partial L}{\partial z_{13}} \\ 
\frac{\partial L}{\partial z_{21}} & \frac{\partial L}{\partial z_{22}} & \frac{\partial L}{\partial z_{23}} \\ 
\vdots & \vdots & \vdots \\ 
\frac{\partial L}{\partial z_{N1}} & \frac{\partial L}{\partial z_{N2}} & \frac{\partial L}{\partial z_{N3}} 
\end{bmatrix}$$

After applying the differentiation for each element, we get the final matrix:

$$\frac{\partial L}{\partial \mathbf{Z}} = \begin{bmatrix} 
-\frac{y_{11}}{z_{11}} & -\frac{y_{12}}{z_{12}} & -\frac{y_{13}}{z_{13}} \\ 
-\frac{y_{21}}{z_{21}} & -\frac{y_{22}}{z_{22}} & -\frac{y_{23}}{z_{23}} \\ 
\vdots & \vdots & \vdots \\ 
-\frac{y_{N1}}{z_{N1}} & -\frac{y_{N2}}{z_{N2}} & -\frac{y_{N3}}{z_{N3}} 
\end{bmatrix}$$

In element-wise notation, this is simply:

$$\left( \frac{\partial L}{\partial \mathbf{Z}} \right)_{ij} = -\frac{y_{ij}}{z_{ij}}$$

