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

from sklearn.model_selection import KFold

lr_grid = [0.005, 0.01, 0.05, 0.1]
bs_grid = [8, 32, 64]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for lr in lr_grid:
    for bs in bs_grid:
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            m = Perceptron(input_dim=2)
            m.fit(X_tr, y_tr, X_val, y_val, epochs=50, lr=lr, batch_size=bs)
            scores.append(np.mean(m.predict(X_val)==y_val))
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        results.append({'lr': lr, 'batch': bs, 'mean_acc': mean_acc, 'std_acc': std_acc})
        print(f"lr={lr:<5.3f} batch={bs:<3} | Acc: {mean_acc:.3f} ± {std_acc:.3f}")

best = max(results, key=lambda x: x['mean_acc'])
print(f"\n✅ Лучшие параметры: lr={best['lr']}, batch={best['batch']} | Acc: {best['mean_acc']:.3f} ± {best['std_acc']:.3f}")

final_model = Perceptron(input_dim=2)
final_model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=best['lr'], batch_size=best['batch'])
print(f"Финальная точность на test: {np.mean(final_model.predict(X_test)==y_test):.2%}")

plt.figure(figsize=(7,4))
plt.plot(final_model.train_loss_history, label='Train')
plt.plot(final_model.val_loss_history, label='Val')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.title('Final Model Training')
plt.show()