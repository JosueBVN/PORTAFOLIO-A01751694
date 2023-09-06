"""
Creado por: Josue Bernardo Villegas Nuño
Matrícula: A01751694
Fecha de entrega: 28/08/2023
Entrega: Momento de Retroalimentación: Módulo 2 Implementación de una técnica
de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
"""

import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Función de activación ReLU
def relu(x):
    return np.maximum(0, x)

# Derivada de ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Cargar datos desde un archivo CSV
def load_csv_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append([float(value) for value in row])
    return np.array(data)

# Coeficiente de determinación (R-squared)
def r_squared(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y_true)**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Cargar datos de entrenamiento y prueba
train_data = load_csv_data('train.csv')
test_data = load_csv_data('test.csv')

# Agregar una columna de ceros a test_data para que tenga la misma estructura que train_data
num_samples_test = test_data.shape[0]
extra_column = np.zeros((num_samples_test, 1))
test_data = np.hstack((extra_column, test_data))

# Dividir el conjunto de datos en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(train_data[:, 1:-1], train_data[:, -1], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalización de características
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

input_size = X_train.shape[1]
hidden_size = 256  # Aumentamos el tamaño de la capa oculta
output_size = 1

np.random.seed(42)
W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)  # Inicialización de He
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)  # Inicialización de He
b2 = np.zeros((output_size, 1))

learning_rate = 0.001  # Mantenemos la tasa de aprendizaje
epochs = 20000  # Aumentamos el número de épocas

# Parámetros adicionales para Adam
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Inicialización de los momentos (Adam)
m_dW1 = np.zeros_like(W1)
v_dW1 = np.zeros_like(W1)
m_db1 = np.zeros_like(b1)
v_db1 = np.zeros_like(b1)
m_dW2 = np.zeros_like(W2)
v_dW2 = np.zeros_like(W2)
m_db2 = np.zeros_like(b2)
v_db2 = np.zeros_like(b2)

# Entrenamiento
for epoch in range(epochs):
    Z1 = np.dot(W1, X_train.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    predictions = Z2

    loss = np.mean((predictions - y_train)**2)

    dZ2 = 2 * (predictions - y_train)
    dW2 = np.dot(dZ2, A1.T) / X_train.shape[0] + 0.01 * W2  # Regularización L2
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X_train.shape[0]
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X_train) / X_train.shape[0] + 0.01 * W1  # Regularización L2
    db1 = np.sum(dZ1, axis=1, keepdims=True) / X_train.shape[0]

    # Actualización de momentos (Adam)
    m_dW1 = beta1 * m_dW1 + (1 - beta1) * dW1
    v_dW1 = beta2 * v_dW1 + (1 - beta2) * dW1**2
    m_db1 = beta1 * m_db1 + (1 - beta1) * db1
    v_db1 = beta2 * v_db1 + (1 - beta2) * db1**2
    m_dW2 = beta1 * m_dW2 + (1 - beta1) * dW2
    v_dW2 = beta2 * v_dW2 + (1 - beta2) * dW2**2
    m_db2 = beta1 * m_db2 + (1 - beta1) * db2
    v_db2 = beta2 * v_db2 + (1 - beta2) * db2**2

    # Corrección de los momentos (Adam)
    m_dW1_corr = m_dW1 / (1 - beta1**(epoch+1))
    v_dW1_corr = v_dW1 / (1 - beta2**(epoch+1))
    m_db1_corr = m_db1 / (1 - beta1**(epoch+1))
    v_db1_corr = v_db1 / (1 - beta2**(epoch+1))
    m_dW2_corr = m_dW2 / (1 - beta1**(epoch+1))
    v_dW2_corr = v_dW2 / (1 - beta2**(epoch+1))
    m_db2_corr = m_db2 / (1 - beta1**(epoch+1))
    v_db2_corr = v_db2 / (1 - beta2**(epoch+1))

    # Actualización de parámetros con momento y adaptación del aprendizaje (Adam)
    W2 -= learning_rate * m_dW2_corr / (np.sqrt(v_dW2_corr) + epsilon)
    b2 -= learning_rate * m_db2_corr / (np.sqrt(v_db2_corr) + epsilon)
    W1 -= learning_rate * m_dW1_corr / (np.sqrt(v_dW1_corr) + epsilon)
    b1 -= learning_rate * m_db1_corr / (np.sqrt(v_db1_corr) + epsilon)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluación en el conjunto de prueba
Z1_test = np.dot(W1, X_test.T) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(W2, A1_test) + b2
predictions_test = Z2_test

r2_score = r_squared(y_test, predictions_test.flatten())

# Binarizar las etiquetas para calcular las métricas
y_test_binary = (y_test > 0.5).astype(int)
predictions_test_binary = (predictions_test.flatten() > 0.5).astype(int)

precision = precision_score(y_test_binary, predictions_test_binary)
recall = recall_score(y_test_binary, predictions_test_binary)
f1 = f1_score(y_test_binary, predictions_test_binary)
confusion = confusion_matrix(y_test_binary, predictions_test_binary)

print(f"R^2 Score on Test Data: {r2_score:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()