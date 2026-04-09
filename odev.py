import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# --- GÖREV 1: VERİ ÖN İŞLEME ---
df = pd.read_csv('Social_Network_Ads.csv')

# Özellik seçimi: Hocanın dediği gibi karar veriyoruz (İkisini de kullanmak en mantıklısı)
selected_features = ['Age', 'EstimatedSalary']
X = df[selected_features].values
y = df['Purchased'].values

# Eğitim ve Test setine ayırma (%75 - %25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling (Ölçeklendirme - Mesafe temelli modellerde şarttır)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- GÖREV 2: LOGISTIC REGRESSION ---
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# --- GÖREV 3: LINEAR REGRESSION ---
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
# Sürekli çıktıları alıyoruz
y_pred_lin_cont = lin_model.predict(X_test_scaled)
# 0.5 threshold (eşik) uygulayarak sınıflandırmaya çeviriyoruz
y_pred_lin = (y_pred_lin_cont >= 0.5).astype(int)

# Metrikleri Hesaplama Fonksiyonu
def calculate_metrics(y_true, y_pred, name):
    print(f"\n--- {name} Performans Sonuçları ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

calculate_metrics(y_test, y_pred_log, "Logistic Regression")
calculate_metrics(y_test, y_pred_lin, "Linear Regression")

# --- GÖREV 5: BONUS GÖRSELLEŞTİRME ---
def plot_decision_boundary(model, X, y, title, ax, is_linear=False):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    if is_linear:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = (Z >= 0.5).astype(int)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel('Yaş (Ölçekli)')
    ax.set_ylabel('Maaş (Ölçekli)')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_boundary(log_model, X_test_scaled, y_test, "Logistic Regression Karar Sınırı", ax1)
plot_decision_boundary(lin_model, X_test_scaled, y_test, "Linear Regression (Threshold 0.5) Karar Sınırı", ax2, is_linear=True)
plt.tight_layout()
plt.show()