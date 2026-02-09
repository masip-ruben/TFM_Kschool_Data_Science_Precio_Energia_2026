TFM – Modelo Predictivo del Precio de la Electricidad en el Mercado Diario del MIBEL

Autores:
- Alejandro Villa Estada
- Elena García Ramírez
- Javier Ijalba Martínez
- Rubén Darío Masip Sánchez

Tutor:
- Antonio Pita Lozano

Descripción del proyecto:
Este repositorio contiene el código y los datos asociados al Trabajo Fin de Máster (TFM) titulado
“Modelo Predictivo del Precio de la Electricidad en el Mercado Diario del MIBEL”, desarrollado en el
Máster en Data Science.

El objetivo principal del proyecto es el desarrollo de un modelo predictivo del precio diario de la
electricidad en el mercado mayorista ibérico (MIBEL), orientado a su integración en modelos financieros
de proyectos de energías renovables. El enfoque del trabajo está alineado con la toma de decisiones
estratégicas y financieras, permitiendo estimar ingresos futuros y apoyar el análisis de rentabilidad
de activos energéticos en un contexto de elevada incertidumbre.

Contexto y caso de uso:
El proyecto se enmarca en el caso de uso de una consultora especializada en análisis energético que
requiere estimaciones robustas del precio de la electricidad para su incorporación en modelos de
project finance. Frente a enfoques deterministas o escenarios únicos, se propone un marco analítico
basado en técnicas de Machine Learning y series temporales que permita capturar la complejidad
del mercado eléctrico.

Alcance del modelo:
El modelo desarrollado estima el precio medio diario de la electricidad bajo un escenario esperado
(P50), utilizando como variables explicativas, entre otras:
- Demanda eléctrica nacional.
- Generación eólica y solar.
- Potencia instalada eólica y solar.
- Variables meteorológicas agregadas a nivel nacional (temperatura, viento y radiación).
- Precio del gas natural.
- Índice de Precios al Consumo (IPC).
- Variables de calendario y festividades.

Para la generación de las variables exógenas futuras, se han desarrollado modelos auxiliares
independientes, lo que permite simular escenarios coherentes a medio y largo plazo.

Metodología:
La arquitectura del proyecto separa claramente la fase de entrenamiento de modelos de la fase de
simulación operativa. Los modelos se entrenan de forma offline a partir de datos históricos y se validan
mediante esquemas de validación temporal. El rendimiento se evalúa frente a un baseline
determinista utilizando métricas como RMSE y WAPE.

El modelo principal de precios se apoya en algoritmos de Machine Learning, integrando las
predicciones generadas por los modelos auxiliares de variables de entrada.

Estructura del repositorio:
- 01.Datos:
  Contiene los datos de entrada y los datos procesados utilizados en el proyecto, procedentes de
  fuentes públicas oficiales del sector energético y climático.
- 02.Modelos:
  Incluye los notebooks y scripts de Python correspondientes a la exploración de datos, generación
  de variables, entrenamiento de modelos auxiliares y desarrollo del modelo final de precios.
- requirements.txt:
  Dependencias del entorno Python.
- requirements-windows.txt:
  Dependencias específicas para entorno Windows.
- README.txt:
  Documento descriptivo del repositorio.

Reproducibilidad:
Los experimentos y modelos incluidos en este repositorio pueden reproducirse a partir de los
ficheros de dependencias proporcionados, utilizando un entorno Python compatible.

Contexto académico:
Este proyecto se ha desarrollado exclusivamente con fines académicos como Trabajo Fin de Máster
en el Máster en Data Science. Los datos empleados proceden de fuentes públicas y no contienen
información confidencial.
