# WoutFrameworkA01751694
# Implementación de Regresión Logística desde Cero

Este proyecto consiste en una implementación manual del algoritmo de Regresión Logística en Python, sin utilizar bibliotecas o frameworks de aprendizaje automático. El objetivo es comprender en profundidad el funcionamiento de la Regresión Logística y cómo se pueden ajustar los parámetros del modelo utilizando el descenso de gradiente.

## Contenido

- [Requisitos](#requisitos)
- [Uso](#uso)
- [Detalles de la Implementación](#detalles-de-la-implementación)
  - [Requisitos](#requisitos-1)
  - [Clase LogisticRegression](#clase-logisticregression)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Requisitos

- Python (3.6 o superior)

## Uso

1. Clona este repositorio en tu máquina local.
2. Ejecuta el archivo `logistic_regression.py`.

## Detalles de la Implementación

La implementación de la Regresión Logística se encuentra en el archivo `logistic_regression.py`. Aquí, se ha creado una clase `LogisticRegression` que contiene métodos para ajustar el modelo y hacer predicciones.

### Clase `LogisticRegression`

- `sigmoid(z)`: Calcula la función sigmoide.
- `fit(X, y)`: Ajusta el modelo de Regresión Logística a los datos de entrenamiento `X` e `y`.
- `predict(X)`: Realiza predicciones binarias sobre los datos de entrada `X`.

La implementación se basa en el descenso de gradiente, donde los parámetros del modelo (weights y bias) se actualizan iterativamente para minimizar la función de costo. Se utiliza la función sigmoide para transformar las salidas del modelo en valores entre 0 y 1, que se interpretan como probabilidades.

## Uso

En el archivo `logistic_regression.py`, puedes ajustar los datos de entrenamiento `X_train` e `y_train` para tus propios datos. Luego, ejecuta el archivo utilizando `python logistic_regression.py`. Las predicciones resultantes se mostrarán en la consola.

Recuerda que esta implementación es básica y no aborda características avanzadas ni consideraciones de optimización. Para aplicaciones reales, se recomienda utilizar bibliotecas como Scikit-Learn, que ofrecen implementaciones robustas y eficientes de algoritmos de aprendizaje automático.

## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar esta implementación o agregar nuevas características, siéntete libre de enviar un pull request.

## Licencia

Este proyecto se encuentra bajo la Licencia MIT.
