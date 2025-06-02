import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Veri Üretimi
fake = Faker()
np.random.seed(42)

n_samples = 200
experience_years = np.random.uniform(0, 10, n_samples)
technical_scores = np.random.uniform(0, 100, n_samples)

# 2. Etiketleme
labels = []
for exp, score in zip(experience_years, technical_scores):
    if exp < 2 and score < 60:
        labels.append(1)  # NOT HIRED
    else:
        labels.append(0)  # HIRED

# 3. Veriyi Hazırlama
X = np.column_stack((experience_years, technical_scores))
y = np.array(labels)

# 4. Veriyi Standardize Edip ve Ayırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Modeli Eğitme
model = SVC(kernel='linear')
model.fit(X_train, y_train)

accuracy =model.score(X_test, y_test)
print(f"Modelin doğruluğu: {accuracy:.2f}")


# 6. Karar Sınırını Görselleştiren Kısım
def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=60, edgecolors='k', alpha=0.7)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=150, linewidth=1.5, facecolors='none', edgecolors='k')

    plt.title("SVM with Faker Data: Hiring Decision Prediction")
    plt.xlabel("Experience Years (standardized)")
    plt.ylabel("Technical Score (standardized)")
    plt.grid(True)
    plt.show()

plot_decision_boundary(model, X_test, y_test)

# 7. Model Performanslarımız
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Tahmin Fonksiyonumuz
def predict_candidate(experience, score):
    input = np.array([[experience, score]])  # bu zaten (1, 2) boyutunda
    input_scaled = scaler.transform(input)   # sadece input ver, köşeli parantez ekleme
    prediction = model.predict(input_scaled)[0]
    if prediction == 0:
        print(f"Experience: {experience}, Score: {score} → HIRED ✅")
    else:
        print(f"Experience: {experience}, Score: {score} → NOT HIRED ❌")


# 9. Örnekler
predict_candidate(1.5, 55)
predict_candidate(5, 70)
predict_candidate(0.5, 40)
predict_candidate(3, 95)
predict_candidate(9, 30)


# 10. Kullanıcıdan Girdi Alarak Tahmin Yapacağımız Kısım
try:
    user_exp = float(input("Tecrübe yılı girin (örn. 3.5): "))
    user_score = float(input("Sınav skoru girin (0-100): "))
    predict_candidate(user_exp, user_score)
except ValueError:
    print("Geçerli bir sayı girmelisiniz.")
