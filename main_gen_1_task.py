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

def run_experiment(param_name, param_values, fixed_params, X_tr, y_tr, X_val, y_val):
    results = []
    for val in param_values:
        print(f"\n--- {param_name} = {val} ---")
        p = Perceptron(input_dim=2)
        if param_name == 'init':
            if val == 'zeros': p.w, p.b = np.zeros((2,1)), 0.0
            elif val == 'large': p.w = np.random.randn(2,1)*10
            else: p.w = np.random.randn(2,1)*0.01
        else:
            fixed_params[param_name] = val
            
        p.fit(X_tr, y_tr, X_val, y_val, **fixed_params)
        acc = np.mean(p.predict(X_val) == y_val)
        results.append({param_name: val, 'Accuracy': acc, 'Final Loss': p.val_loss_history[-1]})
    return pd.DataFrame(results)

print("\n" + "="*50)
print("ЭКСПЕРИМЕНТ 1: Влияние скорости обучения (η)")
print("="*50)
lr_values = [0.001, 0.01, 0.5, 1.0]
plt.figure(figsize=(8, 5))
for lr in lr_values:
    model = Perceptron(input_dim=2)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=lr, batch_size=32)
    test_acc = np.mean(model.predict(X_test) == y_test)
    plt.plot(model.val_loss_history, label=f"η = {lr} (Acc: {test_acc:.2%})")
    print(f"η = {lr:<5} | Точность: {test_acc:.2%}")
plt.xlabel("Эпоха"); plt.ylabel("Validation Loss")
plt.legend(); plt.grid(True); plt.title("Влияние скорости обучения")
plt.show()

print("\n" + "="*50)
print("ЭКСПЕРИМЕНТ 2: Влияние размера батча")
print("="*50)
batch_values = [1, 16, 64, 256]
plt.figure(figsize=(8, 5))
for bs in batch_values:
    model = Perceptron(input_dim=2)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=bs)
    test_acc = np.mean(model.predict(X_test) == y_test)
    plt.plot(model.val_loss_history, label=f"batch = {bs} (Acc: {test_acc:.2%})")
    print(f"batch = {bs:<4} | Точность: {test_acc:.2%}")
plt.xlabel("Эпоха"); plt.ylabel("Validation Loss")
plt.legend(); plt.grid(True); plt.title("Влияние размера батча")
plt.show()

print("\n" + "="*50)
print("ЭКСПЕРИМЕНТ 3: Влияние инициализации весов")
print("="*50)
plt.figure(figsize=(8, 5))

model = Perceptron(input_dim=2)
model.w = np.zeros((2, 1)); model.b = 0.0
model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
acc = np.mean(model.predict(X_test) == y_test)
plt.plot(model.val_loss_history, label=f"Нули (Acc: {acc:.2%})")
print(f"{'Нули':<22} | Точность: {acc:.2%}")

model = Perceptron(input_dim=2)
model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
acc = np.mean(model.predict(X_test) == y_test)
plt.plot(model.val_loss_history, label=f"Маленькие N(0,0.01) (Acc: {acc:.2%})")
print(f"{'Маленькие N(0,0.01)':<22} | Точность: {acc:.2%}")

model = Perceptron(input_dim=2)
model.w = np.random.randn(2, 1) * 10; model.b = 0.0
model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
acc = np.mean(model.predict(X_test) == y_test)
plt.plot(model.val_loss_history, label=f"Большие N(0,10) (Acc: {acc:.2%})")
print(f"{'Большие N(0,10)':<22} | Точность: {acc:.2%}")

plt.xlabel("Эпоха"); plt.ylabel("Validation Loss")
plt.legend(); plt.grid(True); plt.title("Влияние инициализации весов")
plt.show()

# Генераторы
def gen_linear(n=300, sep=2.0):
    X1 = np.random.randn(n//2, 2) + [sep, sep]
    X2 = np.random.randn(n//2, 2) - [sep, sep]
    X = np.vstack([X1, X2])
    y = np.concatenate([np.ones(n//2), np.zeros(n//2)]).reshape(-1, 1)
    return X, y

def gen_xor(n=300):
    X = np.random.randn(n, 2) * 1.5
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int).reshape(-1, 1)
    return X, y

def gen_circle(n=300):
    r = np.random.rand(n) * 2.5
    theta = np.random.rand(n) * 2 * np.pi
    X = np.c_[r * np.cos(theta), r * np.sin(theta)]
    y = (r > 1.2).astype(int).reshape(-1, 1)
    return X, y

def add_noise(X, y, p=0.15):
    mask = np.random.rand(len(y)) < p
    y[mask] = 1 - y[mask]
    return X, y

# Обучение и визуализация
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
datasets = [("Линейно разделимые", gen_linear()), 
            ("XOR (нелинейные)", gen_xor()), 
            ("Окружность (нелинейные)", gen_circle())]

for i, (title, (X_d, y_d)) in enumerate(datasets):
    # Добавляем 10% шума
    X_d, y_d = add_noise(X_d, y_d, p=0.1)
    
    # Стандартизация и split
    X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.3, random_state=42, stratify=y_d)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    y_tr, y_te = y_tr.reshape(-1,1), y_te.reshape(-1,1)
    
    model = Perceptron(input_dim=2)
    model.fit(X_tr, y_tr, X_te, y_te, epochs=100, lr=0.1, batch_size=32)
    acc = np.mean(model.predict(X_te) == y_te)
    
    # Граница
    x_min, x_max = X_te[:,0].min()-0.5, X_te[:,0].max()+0.5
    y_min, y_max = X_te[:,1].min()-0.5, X_te[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    axes[i].contourf(xx, yy, Z, levels=[0,0.5,1], alpha=0.3, cmap='RdBu')
    axes[i].contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)
    axes[i].scatter(X_te[:,0], X_te[:,1], c=y_te.ravel(), cmap='RdBu', edgecolors='k', s=30)
    axes[i].set_title(f"{title}\nAccuracy: {acc:.2%}")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()