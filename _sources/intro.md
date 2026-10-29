# Introducción

*Del análisis de las ventas de Sample Superstore a su proyección*

## El proyecto

Sample Superstore es un conjunto de datos de muestra de una cadena minorista de Estados Unidos
(2014–2017) —una simulación con fines analíticos— con cerca de 10.000 líneas de pedido de muebles,
artículos de oficina y tecnología.

El proyecto tiene dos partes: un análisis exploratorio (EDA) que identifica qué explica el nivel de
ventas, y un modelo de machine learning que usa esos hallazgos para proyectar cuánto venderá la
empresa cada mes, por categoría y región.

## Preguntas del proyecto

1. **¿Qué factores (categoría, región, segmento, tiempo) explican el nivel de ventas (`Sales`)?**
2. **¿Cuánto venderá cada categoría en cada región, mes a mes, durante 2018?**

Los capítulos siguen el flujo del proyecto: planteamiento, datos (ETL y EDA), modelos y resultados,
proyección, recomendaciones aplicadas y conclusiones.

## Contexto

Una cadena minorista vende a través de varias categorías de producto, regiones y segmentos de
cliente, y sus ventas cambian a lo largo del año. Saber qué combinación de factores mueve las ventas,
y anticipar cuánto se venderá, es un insumo directo para planificar: cuánto inventario comprar, qué
metas fijar a cada región y cuándo reforzar la operación.

## Impacto para el negocio

- **Metas comerciales:** fijar la meta de cada región según lo que se espera vender.
- **Inventario y compras:** abastecer cada categoría antes de su temporada alta.
- **Alerta temprana:** detectar dónde se proyecta una caída frente al año anterior.
- **Presupuesto:** proyectar los ingresos del año siguiente.
