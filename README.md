# Grupo 2 · Big Data e IA aplicada

Industria 4.0 · Ingeniería Industrial, FIUNA · Caso Casa Rojas (Fideos Napoli)

Carpeta reproducible del trabajo práctico: limpieza y EDA del dataset, modelos de pronóstico de
demanda y el experimento de madurez de datos que responde la pregunta de investigación.

> ¿Qué nivel de madurez de datos (calidad, frecuencia, historización) requiere una pyme para pasar
> de reportes descriptivos (BI) a modelos predictivos (ML), y qué herramientas open-source permiten
> hacerlo sin licencias costosas?

Los datos **no son de Casa Rojas**: son el dataset público
[Historical Product Demand](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting),
de un fabricante con la misma estructura (planta, depósitos regionales y muchas categorías).

## Notebooks

Se corren en orden. Cada uno lee lo que escribió el anterior.

| Notebook | Responsable | Qué responde | Tiempo |
|---|---|---|---|
| `01_limpieza_eda` | Abi | Qué datos hay, qué tan buenos son y qué ve la empresa con ellos | ~1 min |
| `02_modelo_pronostico` | Franco | Con todos los datos, si el ML le gana a los métodos simples y por cuánto | ~15 min, o ~1 min con la caché |
| `03_experimento_madurez` | Franco | Desde qué historia, calidad y frecuencia vale la pena el ML | ~1 h, o ~1 min con la caché |

`02` y `03` tienen una bandera `RECALCULAR` al principio. En `False` releen los resultados
guardados en `resultados/cache/`, que es lo que conviene para una demo en vivo. Al subir la carpeta
a Drive hay que incluir `resultados/cache/`; sin ella, los notebooks recalculan todo.

## Resultados principales

- **Con todos los datos (02):** LightGBM tiene un WAPE de 35,1 % en 2016, 10 % menos error que la
  media móvil de 12 meses, y le gana los 12 meses. Pronostica 8,8 % de menos, y su rango P10-P90
  cubre el 75,6 % de la demanda real (nominal 80 %).
- **Madurez (03):** el ML vale la pena desde 12 meses de historia; con 6 no. Con un 5 % de
  registros perdidos todavía vale la pena, con un 15 % ya no. Registrar por semana no mejora el
  pronóstico mensual. La media móvil bien elegida consigue el 77 % de la mejora del ML.

La respuesta completa queda en `resultados/tablas/respuesta_pregunta.txt`.

## Cómo correrlo

### En Google Colab

1. Subir esta carpeta entera a Google Drive como `MyDrive/Industria40`.
2. Abrir el notebook desde Drive con Colab.
3. Correr todo. El primer bloque monta Drive, instala `statsforecast` y `kagglehub` si faltan, y
   descarga el dataset sin credenciales de Kaggle.

Si la carpeta tiene otro nombre en Drive, cambiar `RAIZ` en el primer bloque.

### En local

Requiere Python 3.12, la versión de Colab.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks
```

Para ejecutar los tres de punta a punta y guardar las salidas:

```powershell
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\01_limpieza_eda.ipynb notebooks\02_modelo_pronostico.ipynb notebooks\03_experimento_madurez.ipynb
```

## Qué produce

| Carpeta | Contenido | Para quién |
|---|---|---|
| `data/processed/` | Dataset limpio, panel mensual y catálogo de series | `02` y `03` |
| `resultados/powerbi/` | Tablas en estrella listas para importar | Dashboard de Power BI |
| `resultados/tablas/` | Métricas, selección de modelos y criterio del experimento | Informe técnico |
| `resultados/figuras/` | Gráficos en PNG | Informe y slides |

Las columnas de cada tabla de Power BI están en la sección 10 de la
[guía de desarrollo](docs/guia_desarrollo.md).

## Estructura

```
notebooks/        los tres notebooks, ejecutados
src/demanda/      código compartido: datos, variables, modelos, métricas y backtest
data/             crudo (se descarga) y procesado (lo escribe 01); no se versionan
resultados/       figuras, tablas y salidas para Power BI
docs/             guía de desarrollo
```

Las decisiones de modelado, el diseño del experimento y el criterio para decidir si el ML vale la
pena están en [`docs/guia_desarrollo.md`](docs/guia_desarrollo.md).

## Herramientas

Todo es open source y gratuito: Python, pandas, scikit-learn, LightGBM, statsforecast y
matplotlib. El dataset se descarga con `kagglehub`.
