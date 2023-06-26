from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar o dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Pré-processamento: Normalização dos atributos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN com variação de vizinhos
k_values = [1, 3, 5, 7]  # Valores de vizinhos a serem testados

best_k = None
best_knn_score = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    knn_avg_score = knn_scores.mean()

    if knn_avg_score > best_knn_score:
        best_k = k
        best_knn_score = knn_avg_score

# Árvore de Decisão com variação de parâmetros
max_depth_values = [None, 5, 10, 15]  # Valores de profundidade máxima a serem testados

best_max_depth = None
best_dt_score = 0

for max_depth in max_depth_values:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt_scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy')
    dt_avg_score = dt_scores.mean()

    if dt_avg_score > best_dt_score:
        best_max_depth = max_depth
        best_dt_score = dt_avg_score

# SVM com variação de parâmetros
C_values = [0.1, 1.0, 10.0]  # Valores de margem de tolerância a serem testados
kernel_values = ['linear', 'rbf']  # Valores de tipo de kernel a serem testados

best_C = None
best_kernel = None
best_svm_score = 0

for C in C_values:
    for kernel in kernel_values:
        svm = SVC(C=C, kernel=kernel)
        svm_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
        svm_avg_score = svm_scores.mean()

        if svm_avg_score > best_svm_score:
            best_C = C
            best_kernel = kernel
            best_svm_score = svm_avg_score

# Naive Bayes
nb = GaussianNB()
nb_scores = cross_val_score(nb, X_train, y_train, cv=5, scoring='accuracy')
nb_avg_score = nb_scores.mean()

# Treinamento e avaliação dos melhores modelos
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

dt = DecisionTreeClassifier(max_depth=best_max_depth)
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)

svm = SVC(C=best_C, kernel=best_kernel)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)

# Métricas de avaliação
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions)
knn_f1_score = f1_score(y_test, knn_predictions)

dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)
dt_f1_score = f1_score(y_test, dt_predictions)

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
svm_f1_score = f1_score(y_test, svm_predictions)

nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_confusion_matrix = confusion_matrix(y_test, nb_predictions)
nb_f1_score = f1_score(y_test, nb_predictions)

# Curva ROC
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_predictions)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_predictions)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_predictions)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_predictions)

# Plot da Curva ROC
plt.figure()
plt.plot(knn_fpr, knn_tpr, label='KNN')
plt.plot(dt_fpr, dt_tpr, label='Árvore de Decisão')
plt.plot(svm_fpr, svm_tpr, label='SVM')
plt.plot(nb_fpr, nb_tpr, label='Naive Bayes')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

# Impressão dos resultados
print("KNN:")
print("Melhor valor de vizinhos (K):", best_k)
print("Acurácia:", knn_accuracy)
print("Precisão:", knn_precision)
print("Recall:", knn_recall)
print("Matriz de Confusão:")
print(knn_confusion_matrix)
print("F1-Score:", knn_f1_score)
print()

print("Árvore de Decisão:")
print("Melhor valor de profundidade máxima:", best_max_depth)
print("Acurácia:", dt_accuracy)
print("Precisão:", dt_precision)
print("Recall:", dt_recall)
print("Matriz de Confusão:")
print(dt_confusion_matrix)
print("F1-Score:", dt_f1_score)
print()

print("SVM:")
print("Melhor valor de margem de tolerância (C):", best_C)
print("Melhor valor de tipo de kernel:", best_kernel)
print("Acurácia:", svm_accuracy)
print("Precisão:", svm_precision)
print("Recall:", svm_recall)
print("Matriz de Confusão:")
print(svm_confusion_matrix)
print("F1-Score:", svm_f1_score)
print()

print("Naive Bayes:")
print("Acurácia:", nb_accuracy)
print("Precisão:", nb_precision)
print("Recall:", nb_recall)
print("Matriz de Confusão:")
print(nb_confusion_matrix)
print("F1-Score:", nb_f1_score)
print()

# Cálculo das métricas de avaliação
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1_score = f1_score(y_test, knn_predictions)

dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1_score = f1_score(y_test, dt_predictions)

svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1_score = f1_score(y_test, svm_predictions)

nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1_score = f1_score(y_test, nb_predictions)

# Cálculo da média geral para cada método
knn_avg = (knn_accuracy + knn_precision + knn_recall + knn_f1_score) / 4
dt_avg = (dt_accuracy + dt_precision + dt_recall + dt_f1_score) / 4
svm_avg = (svm_accuracy + svm_precision + svm_recall + svm_f1_score) / 4
nb_avg = (nb_accuracy + nb_precision + nb_recall + nb_f1_score) / 4

# Ranking dos métodos
method_scores = {
    "KNN": knn_avg,
    "Árvore de Decisão": dt_avg,
    "SVM": svm_avg,
    "Naive Bayes": nb_avg
}

sorted_scores = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)

print("Ranking dos métodos:")
for method, score in sorted_scores:
    print(f"{method}: {score}")
