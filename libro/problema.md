# Problema y objetivos

*Qué buscamos explicar, qué buscamos predecir y para qué*

## Planteamiento del problema

Las ventas de Superstore varían según qué se vende (categoría), dónde (región), a quién (segmento) y
cuándo (mes y año). Sin entender esos factores, la empresa planifica metas e inventario con base en
intuición, y no puede anticipar los meses de mayor o menor demanda.

**Pregunta 1:** ¿qué factores explican el nivel de ventas (`Sales`)?

**Pregunta 2:** ¿cuánto venderá cada categoría en cada región, mes a mes, durante 2018?

## Objetivo general

Identificar los factores que explican las ventas de Sample Superstore y construir un modelo de
machine learning que proyecte las ventas mensuales por categoría y región para apoyar la
planificación comercial.

## Objetivos específicos

1. Preparar los datos (ETL) y construir la tabla de ventas mensuales por categoría × región.
2. Analizar la distribución de `Sales` y cómo varía según `Category`, `Region`, `Segment` y el
   tiempo (EDA).
3. Entrenar y ajustar 7 modelos de regresión (Ridge, Lasso, KNN, Árbol de decisión, Random Forest,
   XGBoost y SVR).
4. Compararlos entre sí y contra una referencia simple (vender lo mismo que el mismo mes del año
   anterior) con MAE, RMSE y R².
5. Proyectar las ventas de 2018 con el mejor modelo y traducirlas en recomendaciones para el negocio.

## Justificación

Las ventas tienen un patrón estacional claro (suben hacia fin de año) y una tendencia de crecimiento
desde 2016, como muestra el EDA. Ese comportamiento repetible es lo que hace posible proyectarlas. Se
proyecta por categoría y región porque es el nivel al que se fijan metas e inventario, y porque a
nivel de producto o de día los datos son demasiado escasos y ruidosos.
