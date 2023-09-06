# Withouth Framework
# Implementación de Red Neuronal para Regresión

Este proyecto implementa una red neuronal simple para resolver un problema de regresión utilizando solo NumPy. El objetivo es predecir el valor de una variable objetivo (en este caso, "medv") basándose en un conjunto de características.

## Requisitos

Asegúrate de tener instalada la librería NumPy. Puedes instalarla usando el siguiente comando:

```bash
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn

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

## Resultados
A medida que se ajustan los hiperparámetros y se realizan más épocas de entrenamiento, se espera que la pérdida disminuya y que el valor de R^2 Score en el conjunto de prueba aumente. Puede ser necesario ajustar la arquitectura de la red, la tasa de aprendizaje y otros hiperparámetros para obtener un mejor rendimiento.

# WithFrameworkA01751694
## Implementación de Árbol de Decisiones con Python y scikit-learn

Este repositorio contiene una implementación de un modelo de Árbol de Decisiones para la clasificación de datos utilizando Python y la biblioteca scikit-learn. El modelo se entrena y evalúa utilizando el conjunto de datos Iris.

## Contenido del Repositorio

- `codigo.py`: El código principal que implementa el modelo de Árbol de Decisiones, realiza la división de datos, entrenamiento, predicciones y evaluación del modelo. También incluye visualización opcional del árbol de decisiones y la matriz de confusión.

- `README.md`: Este archivo de documentación que proporciona información sobre el contenido y el uso del repositorio.

## Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas de Python en tu entorno antes de ejecutar el código:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Puedes instalar estas bibliotecas utilizando `pip install numpy pandas matplotlib seaborn scikit-learn`.

## Uso

1. Clona este repositorio en tu máquina local:

   ```bash
   git clone https://github.com/tuusuario/turepositorio.git

Abre una terminal o un entorno de desarrollo de Python y navega hasta la carpeta del repositorio.

Ejecuta el archivo codigo.py para entrenar el modelo de Árbol de Decisiones, realizar predicciones y evaluar el rendimiento del modelo. Los resultados se imprimirán en la consola.

    ```bash
    python codigo.py
    ```


## Conjunto de Datos Iris
El conjunto de datos Iris se carga desde scikit-learn y se utiliza para la clasificación de tres especies de flores Iris. El conjunto de datos se divide en un conjunto de entrenamiento y un conjunto de prueba.

## Visualización Opcional
El código incluye visualización opcional del árbol de decisiones y la matriz de confusión utilizando las bibliotecas matplotlib y seaborn. Puedes habilitar o deshabilitar estas visualizaciones editando el código según tus preferencias.

## Contribución
Siéntete libre de contribuir a este repositorio haciendo mejoras o proporcionando ejemplos adicionales. ¡Tus contribuciones son bienvenidas!

## Licencia
Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo LICENSE para obtener más detalles.


## Licencia

Este proyecto se encuentra bajo la Licencia MIT.
