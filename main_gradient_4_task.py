import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

class Perceptron:
    def __init__(self, input_dim):
        self.w = np.random.randn(input_dim, 1) * 0.01
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true, y_pred):
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size):
        self.train_loss_history = []
        self.val_loss_history = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuf, y_shuf = X_train[perm], y_train[perm]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]

                y_pred = self.forward(X_batch)
                
                error = y_pred - y_batch
                grad_w = (X_batch.T @ error) / len(X_batch)
                grad_b = np.mean(error)

                self.w -= lr * grad_w
                self.b -= lr * grad_b

            self.train_loss_history.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_loss_history.append(self.compute_loss(y_val, self.forward(X_val)))

            if (epoch + 1) % 10 == 0:
                acc_t = np.mean((self.forward(X_train) >= 0.5).astype(int) == y_train)
                acc_v = np.mean((self.forward(X_val) >= 0.5).astype(int) == y_val)
                print(f"Epoch {epoch+1:3d} | Train Loss: {self.train_loss_history[-1]:.4f} | Val Loss: {self.val_loss_history[-1]:.4f} | Acc: {acc_t:.2%} / {acc_v:.2%}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

class PerceptronMomentum:
    def __init__(self, input_dim):
        self.w = np.random.randn(input_dim, 1)*0.01; self.b = 0.0
    def sigmoid(self, z): return 1/(1+np.exp(-np.clip(z,-500,500)))
    def forward(self, X): return self.sigmoid(X@self.w + self.b)
    def compute_loss(self, y, p):
        eps=1e-7; p=np.clip(p,eps,1-eps)
        return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
    def fit(self, X_tr, y_tr, X_val, y_val, epochs, lr, batch_size, beta=0.9):
        self.train_loss, self.val_loss = [], []
        v_w, v_b = np.zeros_like(self.w), 0.0
        n = X_tr.shape[0]
        for _ in range(epochs):
            perm = np.random.permutation(n); X_s, y_s = X_tr[perm], y_tr[perm]
            for i in range(0, n, batch_size):
                X_b, y_b = X_s[i:i+batch_size], y_s[i:i+batch_size]
                p = self.forward(X_b); err = p - y_b
                grad_w = (X_b.T @ err)/len(X_b)
                grad_b = np.mean(err)
                v_w = beta*v_w + grad_w
                v_b = beta*v_b + grad_b
                self.w -= lr * v_w; self.b -= lr * v_b
            self.train_loss.append(self.compute_loss(y_tr, self.forward(X_tr)))
            self.val_loss.append(self.compute_loss(y_val, self.forward(X_val)))
    def predict(self, X): return (self.forward(X)>=0.5).astype(int)

plt.figure(figsize=(8,5))
for beta in [0.0, 0.5, 0.9, 0.99]:
    m = PerceptronMomentum(2)
    m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32, beta=beta)
    acc = np.mean(m.predict(X_test)==y_test)
    plt.plot(m.val_loss, label=f"β={beta} (Acc:{acc:.2%})")
plt.xlabel('Epoch'); plt.ylabel('Validation Loss'); plt.legend(); plt.grid(True); plt.title('Momentum SGD')
plt.show()