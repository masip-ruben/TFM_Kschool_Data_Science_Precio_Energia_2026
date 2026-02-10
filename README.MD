# TFM – Modelo Predictivo del Precio de la Electricidad en el Mercado Diario del MIBEL

---

## Autores
- **AVE**
- **EGR**
- **JIM**
- **RDMS**

**Tutor:** APL  
**Máster:** Máster en Data Science  
**Fecha:** Febrero 2026

---
## Motivación
La predicción del precio de la electricidad es un elemento clave en la evaluación financiera de
proyectos de energías renovables. Promotores, fondos de inversión y entidades financieras deben
estimar ingresos futuros en un contexto de elevada incertidumbre, donde el precio eléctrico está
condicionado por múltiples factores como la demanda, el mix de generación, el clima o el precio
de los combustibles fósiles.

Sin embargo, muchos modelos financieros continúan utilizando proyecciones deterministas o
escenarios simplificados, lo que limita su capacidad para capturar la variabilidad real del mercado.
Este proyecto surge con el objetivo de aportar un enfoque analítico más robusto y realista.

---

## Descripción del Proyecto
Este proyecto desarrolla un modelo predictivo del precio diario de la electricidad en el mercado
mayorista ibérico (MIBEL), utilizando técnicas de Machine Learning y modelos de series temporales.

Tras evaluar distintos enfoques de modelización y algoritmos, el modelo seleccionado para la
predicción del precio de la electricidad ha sido **HistGradientBoosting**, al ofrecer el mejor
equilibrio entre precisión, estabilidad y capacidad de generalización.

El modelo principal de precios se apoya, además, en un conjunto de modelos auxiliares destinados a
la generación de variables exógenas futuras, como la generación renovable, la potencia instalada,
la demanda o variables climáticas agregadas a nivel nacional. Este enfoque permite construir
escenarios futuros coherentes y consistentes desde el punto de vista energético.

---

## Potenciales Beneficios para la Empresa
La solución desarrollada permite a empresas del sector energético y financiero:

- Mejorar la estimación de ingresos futuros de proyectos renovables.
- Reducir la dependencia de escenarios únicos o supuestos deterministas.
- Integrar predicciones de precios en modelos financieros para el cálculo de indicadores como
  ingresos esperados, VAN o TIR.
- Apoyar la toma de decisiones estratégicas y de inversión en un entorno de mercado volátil.

---

## Resultados
Los resultados obtenidos muestran que el modelo basado en **HistGradientBoosting** mejora de forma
clara el rendimiento frente a un baseline determinista, capturando con mayor precisión la dinámica
del mercado eléctrico.

La validación del modelo se ha realizado mediante esquemas de partición temporal, respetando la
estructura cronológica de la serie. El desempeño se ha evaluado utilizando el RMSE como métrica
principal, por su interpretación directa en €/MWh, y el WAPE como métrica complementaria, con el
objetivo de analizar el error relativo agregado y verificar la robustez del modelo en distintos
regímenes de precios.

---

## Replicación del Proyecto
El proyecto ha sido diseñado con un enfoque reproducible. Todos los experimentos pueden
replicarse utilizando los datos y el código incluidos en el repositorio, junto con los ficheros de
dependencias proporcionados.

La arquitectura separa la fase de entrenamiento de la fase de simulación operativa, permitiendo
la reutilización de modelos entrenados y la ejecución de nuevas simulaciones bajo distintos
horizontes temporales.
