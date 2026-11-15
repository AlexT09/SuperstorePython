# Protocolo experimental pre-registrado

**Proyecto:** pronóstico mensual de ventas por categoría y región (Sample Superstore).
**Autores:** Alex Teran y David Estrada.
**Fecha de congelamiento:** 2026-11-15. **Etiqueta de Git:** `protocolo-v1`.

Este documento se registró antes de ejecutar `experiments/ejecutar.py`. Todo lo que aparece aquí como
constante está en `experiments/config.py`. Cualquier cambio posterior se declara en el capítulo 6 del libro.

## 1. Lo que ya se conocía al congelar el protocolo

Declararlo es parte del registro: delimita qué resultados son confirmatorios.

- **Sample Superstore ya se había usado para diseñar el modelo.** Hubo tres iteraciones de diseño:
  (1) Random Forest global, (2) perfil estacional común con un peso único elegido en rejilla, y
  (3) el modelo actual, con un peso por serie aprendido por agrupamiento parcial. Se conocían los
  resultados de (1) y (2) con origen móvil en este conjunto (carpeta `paper/`). El modelo (3) no se
  había evaluado en ningún conjunto.
- **Los tres conjuntos adicionales no se habían usado**: se eligieron y descargaron sin ajustar
  ningún modelo sobre ellos. Por eso son la evidencia más limpia de este protocolo.
- Antes del registro solo se hizo una prueba técnica del script (un origen, dos modelos, cuatro
  evaluaciones) para verificar que corría.

## 2. Hipótesis

**Confirmatorias**

- **H1.** En Sample Superstore, el modelo original tiene menor MAE que la mejor combinación de cada
  uno de los 7 modelos base y que las demás referencias.
- **H2.** El modelo original tiene menor MAE que la mejor referencia en al menos 2 de los 3
  conjuntos adicionales y no es significativamente peor que ella en ninguno.

**Exploratorias** (no sustentan por sí solas la superioridad): ablación, sensibilidad a `lam`,
ruido, tamaño de muestra, intervalos de predicción, análisis de errores, RMSE y demás métricas
secundarias, comparación de los métodos de optimización.

## 3. Métricas

- **Principal:** MAE en las unidades originales de cada conjunto, promediado por mes de evaluación
  (un valor por mes: 24 valores por modelo y conjunto).
- **Secundarias:** RMSE, MASE (MAE dividido por el MAE del ingenuo estacional antes del primer
  origen), WAPE y desvío del total (suma pronosticada / suma real − 1).

## 4. Conjuntos de datos

| Conjunto | Fuente | Series | Naturaleza |
|---|---|---|---|
| Sample Superstore (principal) | Tableau, `data/raw/` | 12 (3 categorías × 4 regiones) | ventas minoristas, estacionalidad fuerte |
| Tourism Monthly | Monash / Zenodo, doi:10.5281/zenodo.4656096 | 24 de 366 | demanda turística, estacionalidad muy fuerte |
| Hospital | Monash / Zenodo, doi:10.5281/zenodo.4656014 | 24 de 767 | conteos de pacientes, estacionalidad débil |
| M3 Monthly | Monash / Zenodo, doi:10.5281/zenodo.4656298 | 24 de 1428 | series heterogéneas de seis dominios |

**Criterio de selección de los conjuntos:** repositorio público con licencia abierta, frecuencia
mensual, misma tarea (pronosticar el mes siguiente de un panel de series) y naturaleza distinta en
dominio e intensidad de la estacionalidad. M3 se incluyó a propósito como caso adverso: sus series no
comparten un patrón estacional, que es el supuesto del modelo.

**Criterio de selección de las series** (`experiments/datos_adicionales.py`): series que terminan en
la fecha final más frecuente del conjunto, con los últimos 48 meses completos y mayores que cero; de
ellas, 24 al azar con la semilla 42; se conservan los últimos 48 meses (el largo de Superstore).

## 5. Particiones, repeticiones y semillas

- **Bucle externo:** origen móvil. Los primeros 12 meses solo aportan historia; los siguientes 12 son
  el entrenamiento inicial; los últimos **24 meses** se pronostican uno a uno, reentrenando con todos
  los meses anteriores. Todos los modelos usan exactamente los mismos orígenes y las mismas filas.
- **Bucle interno:** validación cruzada temporal dentro de cada entrenamiento, con pliegues de 3 meses
  y hasta 4 pliegues (`min(4, (meses − 6) // 3)`).
- **Semilla global:** 42 (numpy, scikit-learn, XGBoost, Optuna y DEAP).
- **Modelos con azar** (Random Forest, XGBoost): además se repiten con 10 semillas (0 a 9) con sus
  hiperparámetros elegidos, y se reporta la dispersión. El modelo original es determinista.

## 6. Modelos y presupuesto de ajuste

- **Modelos base (7):** KNN, Ridge, Lasso, árbol de decisión, Random Forest, XGBoost y SVR, dentro de
  un `Pipeline` con one-hot y estandarización.
- **Métodos de optimización (4):** Grid Search, Random Search, bayesiana (Optuna, TPE) y genética (DEAP).
  En Superstore se ejecutan las 7 × 4 = 28 combinaciones. En los conjuntos adicionales se usa la
  búsqueda bayesiana para todos los modelos.
- **Referencias adicionales:** ingenuo estacional y método Theta (sin ajuste) y
  HistGradientBoosting (boosting por histogramas, búsqueda bayesiana).
- **Modelo original:** `src/original_model.py`, hiperparámetro `lam` en [0.01, 1000] (escala log).
  **La combinación que se contrasta es la del modelo original con búsqueda bayesiana**, fijada aquí
  para no elegirla después de ver los resultados.
- **Presupuesto:** 16 evaluaciones para Random, bayesiana y genética, iguales para todos los modelos
  incluido el original. Grid Search evalúa su rejilla completa (entre 4 y 24 combinaciones; 12 para
  el original). Mismos pliegues internos, mismas variables y mismo preprocesamiento para todos.

## 7. Pruebas de hipótesis

Sobre la serie de 24 pérdidas mensuales (MAE del mes) de cada modelo, en cada conjunto:

1. **Model Confidence Set** (Hansen, Lunde y Nason) sobre todos los modelos y combinaciones, con el
   original incluido; estadístico de rango, bootstrap estacionario (5.000 réplicas, bloque medio de 3
   meses), α = 0.10.
2. **Giacomini–White** (instrumentos: constante y diferencia del mes anterior).
3. **Diebold–Mariano** con la corrección de Harvey–Leybourne–Newbold.
4. **Wilcoxon** de rangos con signo (complementaria, no paramétrica).

Las pruebas 2 a 4 comparan el modelo original con **10 referencias**: la mejor combinación de cada
uno de los 7 modelos base (elegida por su MAE del bucle externo, lo que favorece a las referencias),
el ingenuo estacional, Theta y HistGradientBoosting. Los p-valores se ajustan con **Holm** dentro de
cada prueba (10 comparaciones), α = 0.05. Se reportan el p-valor crudo y el ajustado.

- **Intervalos:** bootstrap estacionario del 95 % de la diferencia media de MAE (10.000 réplicas).
- **Tamaño del efecto:** d de Cohen de las diferencias mensuales y delta de Cliff.
- **Entre conjuntos:** Friedman con la corrección de Iman–Davenport sobre el MAE de los modelos (con
  búsqueda bayesiana) en los 4 conjuntos, y Wilcoxon–Holm del original contra cada referencia. Con
  solo 4 conjuntos esta prueba tiene muy poca potencia; se reporta de todos modos.
- **Supuestos:** se reporta Shapiro–Wilk de las diferencias y su autocorrelación (Ljung–Box).

## 8. Regla de decisión

El modelo original se declara **superior** solo si se cumplen las cuatro condiciones:

1. Su MAE medio en Superstore es menor que el de las 10 referencias.
2. Frente a la mejor referencia, al menos tres de las pruebas (MCS, Giacomini–White,
   Diebold–Mariano, Wilcoxon) rechazan la igualdad con p ajustado < 0.05 (para el MCS: la mejor
   referencia queda fuera del conjunto de confianza con α = 0.10).
3. El tamaño del efecto no es despreciable: |d de Cohen| ≥ 0.2 o |delta de Cliff| ≥ 0.147.
4. Tiene menor MAE que la mejor referencia en al menos 2 de los 3 conjuntos adicionales y no es
   significativamente peor (Diebold–Mariano, p ajustado < 0.05) en ninguno.

Si alguna falla, el veredicto es **no superior** y se reporta así.

## 9. Estudios de robustez (exploratorios)

- **Sensibilidad:** MAE con `lam` fijo en 12 valores entre 0.01 y 1000.
- **Semillas:** 10 semillas para Random Forest y XGBoost.
- **Ruido y atípicos:** ruido multiplicativo en la variable objetivo de entrenamiento (desviación
  0.1, 0.2 y 0.4) y 5 % de valores multiplicados por 3; se aplica al original y a las dos mejores
  referencias, con sus hiperparámetros elegidos.
- **Tamaño de muestra:** ventanas de entrenamiento de 12, 18 y 24 meses y subconjuntos de 3, 6 y 12
  series; tiempo de cómputo al replicar los datos.
- **Cambio de distribución:** MAE por separado en los primeros y los últimos 12 meses evaluados, y
  los tres conjuntos adicionales como validación externa.
- **Incertidumbre:** intervalos de predicción conformes. Para cada mes evaluado a partir del 13.º, el
  radio es el cuantil de los errores absolutos del índice en los meses evaluados anteriores,
  multiplicado por el nivel de la serie. Se verifica la cobertura empírica al 80 % y al 90 %.
- **Errores:** por serie, por mes del año y por tercil de ventas.
- **Ablación:** peso único, pesos sin agrupar, promedio simple, solo perfil, perfil por serie y sin
  perfil, con las mismas pruebas (Holm sobre las 6 variantes).
