import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay)

# =============================================================
# 1) VERİ ÖN İŞLEME
# =============================================================
df = pd.read_csv("Social_Network_Ads.csv")

# Gereksiz sütunları kaldır
df = df.drop(["User ID", "Gender"], axis=1)

X = df[["Age", "EstimatedSalary"]].values
y = df["Purchased"].values

# Train-test split (%75 train, %25 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("=== VERİ ÖN İŞLEME ===")
print(f"Toplam örnek   : {len(df)}")
print(f"Train seti     : {len(X_train)}")
print(f"Test seti      : {len(X_test)}")
print(f"Özellikler     : Age, EstimatedSalary")
print(f"Hedef          : Purchased (0/1)\n")

# =============================================================
# 2) LOGISTIC REGRESSION
# =============================================================
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_sc, y_train)
y_pred_lr = lr_model.predict(X_test_sc)

lr_acc   = accuracy_score(y_test, y_pred_lr)
lr_cm    = confusion_matrix(y_test, y_pred_lr)
lr_prec  = precision_score(y_test, y_pred_lr)
lr_rec   = recall_score(y_test, y_pred_lr)
lr_f1    = f1_score(y_test, y_pred_lr)

print("=== LOGİSTİC REGRESSION ===")
print(f"Accuracy  : {lr_acc:.4f}")
print(f"Precision : {lr_prec:.4f}")
print(f"Recall    : {lr_rec:.4f}")
print(f"F1-Score  : {lr_f1:.4f}")
print(f"Confusion Matrix:\n{lr_cm}\n")

# =============================================================
# 3) LINEAR REGRESSION
# =============================================================
lin_model = LinearRegression()
lin_model.fit(X_train_sc, y_train)
y_pred_lin_raw = lin_model.predict(X_test_sc)

# 0.5 threshold ile sınıfa çevir
y_pred_lin = (y_pred_lin_raw >= 0.5).astype(int)

lin_acc   = accuracy_score(y_test, y_pred_lin)
lin_cm    = confusion_matrix(y_test, y_pred_lin)
lin_prec  = precision_score(y_test, y_pred_lin)
lin_rec   = recall_score(y_test, y_pred_lin)
lin_f1    = f1_score(y_test, y_pred_lin)

out_of_range = np.sum((y_pred_lin_raw < 0) | (y_pred_lin_raw > 1))

print("=== LİNEAR REGRESSION (threshold=0.5) ===")
print(f"Accuracy  : {lin_acc:.4f}")
print(f"Precision : {lin_prec:.4f}")
print(f"Recall    : {lin_rec:.4f}")
print(f"F1-Score  : {lin_f1:.4f}")
print(f"Confusion Matrix:\n{lin_cm}")
print(f"\nHam tahmin min : {y_pred_lin_raw.min():.4f}")
print(f"Ham tahmin max : {y_pred_lin_raw.max():.4f}")
print(f"[0,1] dışı tahmin: {out_of_range} / {len(y_pred_lin_raw)}\n")

# =============================================================
# 4) KARŞILAŞTIRMA ANALİZİ
# =============================================================
print("=== KARŞILAŞTIRMA ANALİZİ ===")
print(f"{'Metrik':<12} {'Logistic':>10} {'Linear':>10}")
print("-" * 34)
for name, lv, linv in [
    ("Accuracy",  lr_acc,  lin_acc),
    ("Precision", lr_prec, lin_prec),
    ("Recall",    lr_rec,  lin_rec),
    ("F1-Score",  lr_f1,   lin_f1),
]:
    print(f"{name:<12} {lv:>10.4f} {linv:>10.4f}")

print("""
Analiz Notları:
- Bu veri setinde metrikler aynı çıkmıştır çünkü veri seti küçük
  ve neredeyse doğrusal ayrılabilir yapıdadır.
- Linear Regression çıktıları [0,1] dışına çıkabilir → olasılık yorumu mümkün değil.
- Logistic Regression sigmoid ile çıktıyı [0,1] arasına hapseder.
- Linear Regression bu problem için teorik olarak uygun değildir.
""")

# =============================================================
# 5) GÖRSELLEŞTİRME
# =============================================================
def plot_decision_boundary(model, X_sc, y, title, ax, threshold=None):
    x_min, x_max = X_sc[:, 0].min() - 0.5, X_sc[:, 0].max() + 0.5
    y_min, y_max = X_sc[:, 1].min() - 0.5, X_sc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if threshold is not None:
        Z = (model.predict(grid) >= threshold).astype(int)
    else:
        Z = model.predict(grid)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3,
                colors=["#FFAAAA", "#AAFFAA"])
    ax.contour(xx, yy, Z, colors=["black"], linewidths=1.2)

    for cls, color, label in [(0, "red", "Purchased=0"),
                               (1, "green", "Purchased=1")]:
        mask = y == cls
        ax.scatter(X_sc[mask, 0], X_sc[mask, 1],
                   c=color, label=label, edgecolors="k",
                   linewidths=0.5, s=40, alpha=0.8)

    ax.set_xlabel("Age (standardized)")
    ax.set_ylabel("EstimatedSalary (standardized)")
    ax.set_title(title)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LAB 3 — Logistic vs Linear Regression", fontsize=14, fontweight="bold")

# Decision boundaries — train seti üzerinde
plot_decision_boundary(lr_model,  X_train_sc, y_train,
                       "Logistic Regression — Train", axes[0, 0])
plot_decision_boundary(lin_model, X_train_sc, y_train,
                       "Linear Regression — Train",   axes[0, 1], threshold=0.5)

# Confusion matrices — test seti
ConfusionMatrixDisplay(lr_cm,  display_labels=["Satın almadı","Satın aldı"]).plot(
    ax=axes[1, 0], colorbar=False)
axes[1, 0].set_title("Confusion Matrix — Logistic Regression")

ConfusionMatrixDisplay(lin_cm, display_labels=["Satın almadı","Satın aldı"]).plot(
    ax=axes[1, 1], colorbar=False)
axes[1, 1].set_title("Confusion Matrix — Linear Regression")

plt.tight_layout()
plt.savefig("lab3_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Görsel kaydedildi: lab3_results.png")
