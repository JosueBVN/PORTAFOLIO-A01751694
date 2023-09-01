# Implementación de Red Neuronal para Regresión

Este proyecto implementa una red neuronal simple para resolver un problema de regresión utilizando solo NumPy. El objetivo es predecir el valor de una variable objetivo (en este caso, "medv") basándose en un conjunto de características.

## Requisitos

Asegúrate de tener instalada la librería NumPy. Puedes instalarla usando el siguiente comando:

```bash
pip install numpy
pip install scikit-learn

```
## Dataset
El dataset utilizado se divide en dos archivos CSV: train.csv y test.csv. El archivo train.csv contiene los datos de entrenamiento y el archivo test.csv contiene los datos de prueba.

## Uso
El código principal se encuentra en el archivo main.py. A continuación se describen los pasos clave del proceso de implementación:

## Carga de Datos
Los datos de entrenamiento y prueba se cargan desde los archivos CSV utilizando la función load_csv_data.

Normalización de Características: Las características se normalizan restando la media y dividiendo por la desviación estándar de los datos de entrenamiento.

## Arquitectura de la Red Neuronal
 La red neuronal tiene dos capas ocultas. Puedes ajustar el tamaño de las capas y otros hiperparámetros en el código.

## Entrenamiento
Se realizan múltiples épocas de entrenamiento. El algoritmo utiliza el algoritmo de optimización Adam con momentos y adaptación de la tasa de aprendizaje.

## Predicciones y Evaluación
Se calculan las predicciones en el conjunto de prueba y se evalúan utilizando el coeficiente de determinación (R^2 Score).

##Resultados
A medida que se ajustan los hiperparámetros y se realizan más épocas de entrenamiento, se espera que la pérdida disminuya y que el valor de R^2 Score en el conjunto de prueba aumente. Puede ser necesario ajustar la arquitectura de la red, la tasa de aprendizaje y otros hiperparámetros para obtener un mejor rendimiento.

## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar esta implementación o agregar nuevas características, siéntete libre de enviar un pull request.

## Licencia

Este proyecto se encuentra bajo la Licencia MIT.
