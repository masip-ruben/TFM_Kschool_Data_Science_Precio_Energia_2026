# TFM – Modelo Predictivo del Precio de la Electricidad en el Mercado Diario del MIBEL

## Autores
- **Alejandro Villa Estada**
- **Elena García Ramírez**
- **Javier Ijalba Martínez**
- **Rubén Darío Masip Sánchez**

**Tutor:** Antonio Pita Lozano  
**Máster:** Máster en Data Science  
**Fecha:** Febrero 2026

---

## Descripción del proyecto
Este repositorio contiene el código y los datos asociados al Trabajo Fin de Máster (TFM) titulado  
**“Modelo Predictivo del Precio de la Electricidad en el Mercado Diario del MIBEL”**.

El objetivo principal del proyecto es desarrollar un **modelo predictivo del precio diario de la electricidad**
en el mercado mayorista ibérico (MIBEL), orientado a su **integración en modelos financieros de proyectos
de energías renovables**, apoyando la toma de decisiones estratégicas y de inversión en un entorno de
alta incertidumbre.

---

## Contexto y caso de uso
El trabajo se enmarca en el caso de uso de una **consultora especializada en análisis energético** que
requiere estimaciones robustas del precio de la electricidad para su incorporación en modelos de
*project finance*. Frente a enfoques deterministas o escenarios únicos, se propone un marco analítico
basado en **Machine Learning y series temporales** capaz de capturar la complejidad del mercado eléctrico.

---

## Alcance del modelo
El modelo estima el **precio medio diario de la electricidad** bajo un escenario esperado (P50),
utilizando como variables explicativas:

- Demanda eléctrica nacional  
- Generación eólica y solar  
- Potencia instalada eólica y solar  
- Variables meteorológicas agregadas a nivel nacional (temperatura, viento y radiación)  
- Precio del gas natural  
- Índice de Precios al Consumo (IPC)  
- Variables de calendario y festividades  

Para la generación de variables exógenas futuras se han desarrollado **modelos auxiliares independientes**,
permitiendo simular escenarios coherentes a medio y largo plazo.

---

## Metodología
La arquitectura del proyecto separa claramente:

- **Fase de entrenamiento**:  
  Ingestión de datos históricos, preprocesado, generación de variables y entrenamiento de modelos.
- **Fase de simulación operativa**:  
  Uso de modelos ya entrenados para la generación de escenarios futuros.

La validación se realiza mediante **esquemas temporales** y el rendimiento se evalúa frente a un
**baseline determinista**, utilizando métricas como **RMSE** y **WAPE**.

El **modelo principal de precios** se apoya en algoritmos de *Machine Learning*, integrando como entrada
las predicciones generadas por los modelos auxiliares.

---

## Estructura del repositorio

- 01.Datos
- 02.Modelos
- requirement.txt
- requirement-windows.txt
- README.MD

- **01.Datos/**  
  Contiene los conjuntos de datos utilizados en el proyecto, obtenidos a partir de fuentes públicas
  del sector energético y climático.

- **02.Modelos/**  
  Incluye los notebooks y scripts desarrollados para el análisis de datos, la generación de variables
  y la construcción de los modelos predictivos.

- **requirements.txt**  
  Fichero con las dependencias necesarias para reproducir el proyecto en un entorno Python.

- **requirements-windows.txt**  
  Fichero con dependencias específicas para la ejecución del proyecto en entornos Windows.

- **README.md**  
  Documento descriptivo del repositorio y del alcance del proyecto.


---

## Reproducibilidad
Los experimentos y modelos incluidos en este repositorio pueden reproducirse **a partir de los ficheros
de dependencias**, utilizando un entorno Python compatible.

---

## Contexto académico
Este proyecto se ha desarrollado **exclusivamente con fines académicos** como Trabajo Fin de Máster
en el Máster en Data Science.  
Todos los datos empleados proceden de **fuentes públicas oficiales** del sector energético y climático.
