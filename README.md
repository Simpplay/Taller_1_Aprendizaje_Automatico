# Taller 1 — Aprendizaje Automático (2026-2S)

**Datos, modelos lineales y decisiones bajo incertidumbre.** Entrega: 8 de octubre de 2026.
Autor: **Juan José Jiménez Maya** (entrega individual).

Esta entrega incluye el Problema 1 (teórico) y los Problemas 4, 5, 6 y 7 (prácticos). Los Problemas 2 y 3 no se presentan.

## Contenido

```
mlcore.py                              mi módulo: OLS, Ridge, descenso por gradiente lineal/logístico y Lasso proximal
01_implementacion_y_simulacion.ipynb   problemas 4 y 5 (incluye las 30 pruebas automáticas del 4b)
02_regresion_airfoil.ipynb             problema 6
03_clasificacion_vuelos.ipynb          problema 7
data/airfoil_self_noise.dat            copia local de UCI (si falta, el notebook 02 la descarga)
resultados/                            tablas (CSV/JSON) y figuras (PNG) que generan los notebooks
informe/informe.tex, informe.pdf       informe (pdflatex)
requirements.txt                       versiones usadas
```

## Cómo ejecutarlo

Desde la carpeta de la entrega, con Python instalado:

```bash
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace 01_implementacion_y_simulacion.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace 02_regresion_airfoil.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace 03_clasificacion_vuelos.ipynb
cd informe
pdflatex informe.tex
pdflatex informe.tex
```

* **Orden:** 01 → 02 → 03. Los tres importan `mlcore.py` y no dependen uno del otro.
* **Tiempo medido** (última ejecución en Linux, Python 3.13, `jupyter nbconvert --execute` desde un kernel limpio):
  **01 ≈ 24 s, 02 ≈ 12 s, 03 ≈ 24 s** (≈ 1 min en total).
* **Informe:** compila con `pdflatex` (11 páginas, incluida la bibliografía). Las figuras se leen de `../resultados/`.
* **Reproducibilidad:** al borrar `resultados/` y volver a ejecutar los notebooks se regeneran las mismas tablas.
  Entre Windows y Linux solo cambian las últimas cifras decimales de algunos números (redondeo de punto flotante);
  ninguna decisión, conteo ni valor del informe cambia.

## Datos y semillas

| Caso | Origen | Semillas |
|---|---|---|
| Airfoil Self-Noise | UCI ML Repository, Brooks, Pope y Marcolini (1989), doi:10.24432/C5VW2C, CC BY 4.0; misma URL que el notebook de clase. | Partición 80/20 `random_state=2026`; `KFold(5, shuffle=True, random_state=2026)` |
| Vuelos sintéticos | `generar_vuelos` copiada sin cambios de `Taller_clasificacion_vuelos.ipynb` | `generar_vuelos(n=9600, seed=2026)`; partición temporal fija |
| Simulación P5 | Generador del enunciado | Semilla maestra 2026 → subsemillas en `resultados/p5_subsemillas.json` |
| Pruebas P4 | Datos sintéticos | 2026 (+0…6 según la prueba) |

## Resultados guardados (`resultados/`)

| Archivo | Contenido |
|---|---|
| `p4_pruebas_numericas.csv`, `p4_contraste_sklearn.csv` | 30 pruebas automáticas y comparación con scikit-learn (con N y N/2) |
| `p5_sesgo_varianza.csv`, `p5_ridge_teoria_vs_empirica.csv` | Sesgo, varianza y error por escenario, modelo y λ; Ridge teórico vs. empírico |
| `p5_lasso_seleccion.csv`, `p5_lasso_frecuencias.csv`, `p5_subsemillas.json` | Selección de Lasso y semillas |
| `p6_auditoria_*.csv`, `p6_particiones.csv`, `p6_pliegues_cv.csv` | Auditoría y particiones por `id` |
| `p6_cv_por_pliegue.csv`, `p6_cv_resumen.csv`, `p6_registro_seleccion.json` | Validación cruzada y selección |
| `p6_evaluacion_final.csv`, `p6_errores_por_segmento.csv`, `p6_predicciones_prueba.csv` | Evaluación final y predicciones |
| `p7_particiones.csv`, `p7_cv_por_pliegue.csv`, `p7_cv_resumen.csv`, `p7_historial_seleccion.json` | Partición temporal, CV y selección |
| `p7_contraexperimento_fuga.csv`, `p7_diagnostico_validacion.csv`, `p7_calibracion_validacion.csv` | Fuga y diagnóstico de probabilidades |
| `p7_tabla_umbrales_validacion.csv`, `p7_manifiesto_politica.json` | Tabla de umbrales y manifiesto guardado antes de abrir prueba |
| `p7_evaluacion_final_prueba.csv`, `p7_predicciones_prueba.csv` | Reporte final y predicciones (id, fecha, etiqueta, probabilidad, decisiones) |
| `p*.png` | Figuras del informe |

## Atribución

El detalle función por función está en la sección «Atribución» del informe.

* De `Regresion_lineal_logistica_regularizacion.ipynb`: `LinearRegressionScratch` → `LinearRegressionGD`; `LogisticRegressionScratch` → `LogisticRegressionGD`; `fit_regularized` → `gradiente_proximal` + `LassoRegression`; `sigmoid`/`binary_cross_entropy` → `sigmoid`, `logistic_loss_terms`; `prepare_data` y la celda de descarga → `ajustar_predecir` y la carga del notebook 02. Eliminé el parámetro de tasa manual (`learning_rate`) porque siempre uso η = 1/L.
* De `Taller_clasificacion_vuelos.ipynb`: `generar_vuelos` (literal); `separar_por_tiempo`, `construir_pipeline` (incluido `LogisticRegression(class_weight='balanced')` para el modelo ponderado), `comparar_fuga`, `tabla_umbrales`/`seleccionar_umbral`, `metricas_manuales`/`comparar_politicas` → celdas equivalentes del notebook 03.
* Del notebook `Clase_practica_EDA_para_Machine_Learning.ipynb`: el flujo del EDA del notebook 02 (hipótesis, univariado, asociación, grupos con tamaño, regla IQR solo para diagnóstico).

## Uso de IA generativa

Utilicé IA generativa como apoyo para encaminar las demostraciones (redacté el planteamiento original en Obsidian y utilicé Claude exclusivamente para transcribir y formatear el texto de Markdown a LaTeX), y también la utilicé para adaptar lo necesario del código base de los talleres a las necesidades y requerimientos específicos de esta implementación.
