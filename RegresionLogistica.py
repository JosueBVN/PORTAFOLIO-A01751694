"""
Creado por : Josue Bernardo Villegas Nuño
Matricula : A01751694
Fecha de entrega: 28/08/2023
Entrega : Momento de Retroalimentación: Módulo 2 Implementación de una técnica
 de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
"""

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            
            # Gradientes
            dw = (1/num_samples) * np.dot(X.T, (predictions - y))
            db = (1/num_samples) * np.sum(predictions - y)
            
            # Actualización de parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X, threshold=0.5):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        binary_predictions = [1 if p >= threshold else 0 for p in predictions]
        return binary_predictions

# Datos de entrenamiento
X_train = np.array([[2.3, 4.5], [1.5, 3.2], [3.9, 7.5], [2.5, 5.8]])
y_train = np.array([0, 0, 1, 1])

# Crear modelo y entrenar
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)

# Datos de prueba
X_test = np.array([[2.7, 5.3], [1.8, 3.7]])
predictions = model.predict(X_test)

print("Predictions:", predictions)
