


# Metodología

## 1. Datos

Se utilizan datos históricos previamente procesados para obtener
rendimientos logarítmicos.

## 2. Modelado

Los rendimientos históricos se utilizan para estimar los
parámetros del modelo estadístico.

## 3. Monte Carlo

Se generan múltiples trayectorias posibles utilizando los
parámetros estimados.

## 4. Ejecución secuencial

Los escenarios se procesan mediante una implementación
secuencial que constituye la línea base.

## 5. Ejecución paralela

Los escenarios independientes se distribuyen entre múltiples
procesos.

## 6. Benchmark

Se comparan diferentes cantidades de escenarios y configuraciones
de procesamiento.

## 7. Riesgo

Los resultados se analizan mediante métricas estadísticas y
financieras.

## 8. Adaptación

Se estudia un criterio de convergencia estadística que permite
determinar cuándo el número de escenarios puede considerarse
suficiente.

## 9. Reproducibilidad

Los experimentos utilizan semillas controladas cuando corresponde
y los resultados de benchmarks son obtenidos mediante ejecución real.