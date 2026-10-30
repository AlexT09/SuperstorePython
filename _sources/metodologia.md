# Metodología

*Cómo se pasa de los datos crudos a la proyección*

## Flujo del proyecto

1. **ETL:** CSV crudo → dataset limpio y tabla mensual (categoría × región × mes).
2. **EDA:** distribución de las ventas, factores, estacionalidad y tendencia.
3. **Variables:** mes del año e historial de ventas, sin información del futuro.
4. **Modelos:** 7 regresores dentro de un Pipeline.
5. **Ajuste:** Grid Search con validación cruzada temporal.
6. **Evaluación:** 2017 reservado: MAE, RMSE y R² frente a la referencia.
7. **Proyección:** mejor modelo reentrenado con 2015–2017 → 2018 mes a mes.

## Datos y unidad de análisis

- Se agregan las 9.994 líneas de pedido en ventas mensuales por categoría × región: 12 series × 48
  meses = 576 filas.
- 2014 solo aporta historia (las variables necesitan 12 meses previos).
- Entrenamiento: **2015–2016** (288 filas). Prueba: **2017** (144 filas), reservado y usado una sola
  vez.
- La división es por fecha, no al azar: mezclar meses dejaría al modelo ver el futuro (data leakage).

## Variable objetivo

El modelo predice un índice: cuántas veces el promedio de los últimos 12 meses se venderá ese mes.
Luego se multiplica por ese promedio para volver a dólares.

Así una misma regla sirve para series grandes (Technology en West) y pequeñas (Furniture en South), y
la proyección sigue el nivel reciente de cada serie. Con las ventas en dólares directamente, la
proyección anual se desviaba hasta un 38 %.

## Pipeline y validación

- Pipeline de scikit-learn: `OneHotEncoder` (categoría, región) + `StandardScaler` (numéricas) +
  modelo. El preprocesamiento se ajusta solo con datos de entrenamiento en cada pliegue.
- `TimeSeriesSplit` con 4 pliegues: cada uno entrena con los meses anteriores y valida con los 3
  meses siguientes.
- Semilla fija (42) para que los resultados se puedan reproducir.

## Búsqueda de hiperparámetros

- **Grid Search:** prueba todas las combinaciones de una rejilla de hiperparámetros definida para
  cada modelo.
- Criterio: menor MAE en validación cruzada. El mejor modelo se elige con ese error, no con el de
  2017, para que la prueba siga siendo independiente.
- Referencia: proyectar lo mismo que el mismo mes del año anterior. Un modelo solo aporta si la
  supera.
