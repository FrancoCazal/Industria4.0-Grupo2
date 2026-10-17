# Guía de desarrollo · repositorio del Grupo 2

Industria 4.0 · Ingeniería Industrial, FIUNA · Big Data e IA aplicada · Caso Casa Rojas (Fideos Napoli)

Este documento define qué se construye en este repositorio, con qué decisiones y cuándo está
terminado. Es la referencia para implementar y para revisar: si el código y la guía no coinciden,
se corrige uno de los dos en el mismo cambio.

## 1. Alcance

El repo es la **carpeta reproducible** que pide el TP (sección 7): código, datos y resultados del
pipeline, el modelo y el experimento de madurez.

| Entra | Queda fuera |
|---|---|
| Ingesta, limpieza, calidad y EDA del dataset | El archivo `.pbix` de Power BI (lo arma Sofi con las salidas del repo) |
| Modelos de pronóstico, validación y métricas | El informe técnico, las slides y la ficha de caso |
| Experimento de madurez (historia, faltantes, frecuencia) | Cualquier cosa de Blindside |
| Tablas y figuras para el informe, las slides y Power BI | Despliegue, API o interfaz |

## 2. La pregunta y cómo la responde el repo

> ¿Qué nivel de madurez de datos (calidad, frecuencia, historización) requiere una pyme para pasar
> de reportes descriptivos (BI) a modelos predictivos (ML), y qué herramientas open-source permiten
> hacerlo sin licencias costosas?

| Parte de la pregunta | Dónde se responde | Con qué evidencia |
|---|---|---|
| Calidad | `01` y `03` | Diagnóstico ISO/IEC 25012 del dataset; experimento con 5, 15 y 30 % de registros perdidos |
| Historización | `03` | El mismo backtest con 6, 12, 24 y 36 meses de historia, y la historia completa |
| Frecuencia | `03` | El mismo pronóstico mensual hecho con datos mensuales y con datos semanales |
| BI → ML | `02` y `03` | Si el ML le gana a los métodos simples, por cuánto y desde qué nivel de madurez |
| Herramientas open-source | Todo el repo | Python, pandas, scikit-learn, LightGBM y statsforecast, sin licencias |

## 3. Dataset

[Historical Product Demand](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)
(Kaggle, `felixzhao/productdemandforecasting`). Demanda de pedidos de un fabricante con 4 depósitos
centrales. La empresa del dataset necesita pronosticar la demanda **del mes subsiguiente**, porque sus
productos tardan más de un mes en llegar a los depósitos. Es el mismo problema de una fábrica
paraguaya con insumos importados y distribución al interior.

Se descarga con `kagglehub` sin credenciales y no se versiona.

Hechos verificados sobre el archivo crudo (el notebook `01` los recalcula y no los copia):

| Hecho | Valor |
|---|---|
| Filas | 1.048.575, exactamente el máximo de una hoja de Excel sin el encabezado |
| Columnas | `Product_Code`, `Warehouse`, `Product_Category`, `Date`, `Order_Demand` |
| Productos, depósitos y categorías | 2.160, 4 y 33; ningún producto cambia de categoría |
| Series producto × depósito | 2.849 |
| Rango de fechas | 2011-01-08 a 2017-01-09; 2011 tiene 640 filas y 2017 tiene 53 |
| Filas sin fecha | 11.239, todas del depósito `Whse_A` |
| Demanda guardada como texto | Sí; los negativos vienen entre paréntesis, por ejemplo `(100)` |
| Filas negativas | 10.469 (probables devoluciones) |
| Filas con demanda cero | 28.681 |
| Filas duplicadas exactas | 122.423; sin número de pedido no se puede distinguir un duplicado de dos pedidos iguales |
| Peso del depósito `Whse_J` | 73 % de las filas y 65 % del volumen |

## 4. Estructura del repositorio

```
Industria40/
├── README.md                    cómo correrlo, en local y en Colab
├── requirements.txt             versiones fijas, Python 3.12
├── .gitignore
├── docs/
│   └── guia_desarrollo.md       este documento
├── src/demanda/                 código compartido por los notebooks
│   ├── config.py                rutas, semilla, fechas del backtest
│   ├── datos.py                 descarga, lectura, limpieza y armado del panel
│   ├── features.py              variables ancladas en el origen
│   ├── modelos.py               referencias, ETS, Ridge, LightGBM y logística
│   ├── metricas.py              MAE, WAPE, sesgo, MASE y clasificación
│   ├── backtest.py              origen móvil y escenarios del experimento
│   └── graficos.py              estilo común de las figuras
├── notebooks/
│   ├── 01_limpieza_eda.ipynb            escalón 2: datos
│   ├── 02_modelo_pronostico.ipynb       escalón 3: modelo
│   └── 03_experimento_madurez.ipynb     escalón 3 y cierre: la respuesta
├── data/
│   ├── raw/                     CSV descargado (no se versiona)
│   └── processed/               salidas de 01, entradas de 02, 03 y Power BI
└── resultados/
    ├── figuras/                 PNG para el informe y las slides
    ├── tablas/                  CSV de métricas, seleccion.json y la respuesta del experimento
    ├── powerbi/                 tablas listas para importar en Power BI
    └── cache/                   predicciones de los backtests (no se versiona)
```

### ¿Alcanza con un notebook?

Técnicamente sí, pero se separa en tres por cuatro razones:

1. **Cada notebook es un escalón de la exposición.** `01` respalda el escalón de datos, `02` el
   modelo y `03` la respuesta. Los notebooks los muestra Franco; Power BI lo muestra Abi con las
   tablas de `resultados/powerbi/`.
2. **La frontera del plan es un archivo.** «Hasta el dataset limpio» y «desde feature engineering»
   se traduce en que `01` escribe `data/processed/` y `02` y `03` lo leen. Power BI lee lo mismo.
3. **El experimento es lento** (del orden de 45 minutos, casi todo ETS) y la demo en vivo
   tiene que correr en un minuto. `02` y `03` guardan sus resultados en `resultados/cache/` y se
   pueden releer sin recalcular.
4. **El código repetido va a `src/demanda/`.** Métricas, modelos y backtest los usan `02` y `03`;
   duplicarlos haría que los números de los dos notebooks dejen de coincidir.

Regla de los notebooks: el razonamiento, los gráficos y las conclusiones viven en el notebook; la
lógica reutilizable vive en `src/demanda/`. La limpieza queda escrita **dentro** de `01` porque es
lo que se muestra en vivo, y `datos.py` tiene la misma limpieza como función para que `03` la
pueda repetir sobre datos degradados. Un `assert` en `01` verifica que las dos den el mismo
resultado.

## 5. Entorno

- **Python 3.12**, la versión de Colab. `requirements.txt` fija versiones exactas.
- En **Colab**, la carpeta del repo se sube a Google Drive y el primer bloque de cada notebook monta
  Drive y apunta a esa carpeta. Colab ya trae pandas, scikit-learn, LightGBM y matplotlib; el
  notebook instala `statsforecast` y `kagglehub` solo si faltan.
- En **local**: `py -3.12 -m venv .venv`, `pip install -r requirements.txt` y Jupyter.
- Cada notebook imprime las versiones de sus librerías al empezar.
- Semilla única: `SEMILLA = 42`, en `config.py`.

## 6. Decisiones de modelado

### 6.1 Unidad de pronóstico y horizonte

- **Serie:** producto × depósito. Son 2.849 series.
- **Frecuencia de la decisión:** mensual.
- **Horizonte:** con los datos hasta el cierre del mes `t`, se pronostica el mes `t + 2`, que es
  el problema que declara la empresa del dataset. Para el caso, equivale a decidir hoy la
  producción que va a llegar a Encarnación o a Ciudad del Este dentro de dos meses.

### 6.2 Panel mensual

- Período: **enero de 2012 a diciembre de 2016**, 60 meses completos. 2011 y 2017 se excluyen
  por incompletos y se documenta cuántas filas se pierden.
- **Demanda** = suma de los pedidos positivos del mes. Los negativos se tratan como devoluciones:
  no se restan de la demanda y se reportan en una columna aparte.
- Las filas sin fecha se descartan, porque no se pueden ubicar en el tiempo, y se documenta su
  peso en filas y en volumen.
- Un mes sin pedidos vale **cero** desde el primer mes con demanda de la serie. Antes de ese mes,
  la serie no existe y el valor es nulo, no cero.
- Los duplicados exactos se conservan como pedidos distintos y la decisión se declara.

### 6.3 Validación

**Origen móvil** (Tashman, 2000). En cada origen `t` cada modelo se entrena solo con lo conocido
hasta `t` y pronostica `t + 2`.

| Uso | Orígenes | Meses pronosticados | Para qué |
|---|---|---|---|
| Selección | 6 | 2015-02 a 2015-12, uno cada dos meses | Elegir la ventana de la media móvil y la pérdida de LightGBM |
| Prueba (`02`) | 12 | 2016-01 a 2016-12 | Los resultados que se reportan |
| Experimento (`03`) | 6 | 2016-02 a 2016-12, uno cada dos meses | Los escenarios de madurez |

Lo que se elige en 2015 no se vuelve a tocar al ver 2016; `02` lo guarda en
`resultados/tablas/seleccion.json` y `03` lo lee de ahí. El experimento usa la mitad de los
orígenes de la prueba porque ETS cuesta cerca de un minuto por origen y hay nueve escenarios.

### 6.4 Métricas

| Métrica | Definición | Para qué |
|---|---|---|
| MAE | Error absoluto medio, en unidades | Lectura directa |
| **WAPE** | Σ\|error\| / Σ demanda | **Principal**, ponderada por volumen |
| Sesgo | (Σ pronóstico − Σ demanda) / Σ demanda | Si el modelo pide de más o de menos |
| MASE | Error absoluto / MAE dentro de la historia del naive de un paso (Hyndman y Koehler, 2006), promediado sobre series | Pesa igual a todas las series, grandes y chicas |
| Mejora vs referencia | 1 − WAPE del modelo / WAPE de la referencia | «Mejoró X % sobre la media móvil» |

- Toda métrica se reporta con **promedio, desvío y peor origen**; nunca como un número suelto.
- La escala de MASE se calcula siempre con la historia **completa** hasta el origen, también en
  los escenarios con historia recortada, para que la vara sea la misma en todos los escenarios.
  Las series con escala cero (historia constante) quedan fuera del MASE y se cuentan.
- Un naive con MASE mayor que 1 es esperable: el denominador es un error de un paso dentro de la
  historia y el numerador un error a dos meses fuera de ella. El notebook lo tiene que explicar.
- **No se usa MAPE**: divide por la demanda de cada mes y no se puede calcular cuando vale cero.
- Clasificación (anexo de `02`): precisión, recall y F1 de la clase rara, al lado del accuracy
  del clasificador trivial.

### 6.5 Modelos

| Modelo | Familia | Cómo pronostica `t + 2` |
|---|---|---|
| Naive | Referencia | El valor de `t` |
| Naive estacional | Referencia | El mismo mes del año anterior, `t − 10`. No aplica con menos de 11 meses de historia |
| Media móvil | Referencia, la regla de la pyme | Promedio de los últimos `k` meses; `k` entre 3, 6 y 12 se elige en la selección |
| Croston SBA | Clásico, demanda intermitente | Tamaño e intervalo entre pedidos, suavizados por separado (α = 0,1) |
| ETS | Estadístico, por serie | `AutoETS` de statsforecast; estacional (12) con 24 meses o más, sin estacionalidad con menos de 24, y no aplica con menos de 12 |
| Ridge | ML lineal, global | Mismas variables que LightGBM; nulos imputados con indicador, variables estandarizadas |
| LightGBM | ML con árboles, global | Un solo modelo para todas las series |
| Regresión logística | Clasificación (anexo) | ¿Habrá algún pedido en `t + 2`? |

Los modelos **globales** se entrenan con todas las series a la vez. Ninguno usa el código de
producto como variable, para no memorizar productos. Todo pronóstico negativo se lleva a cero. Una
serie sin historia en el origen recibe pronóstico cero en todos los modelos.

Cuando un modelo no se puede aplicar a una serie (el naive estacional sin un año de historia, ETS
con menos de 12 períodos), esa serie recibe la media móvil elegida y se registra la fracción. Si un
modelo no se pudo aplicar a la mitad de las series de un escenario, no se reporta en ese escenario:
sus cifras serían las de la media móvil.

**Configuración de LightGBM**, fija y conservadora: 300 árboles, tasa de aprendizaje 0,03, 15 hojas,
al menos 200 ejemplos por hoja y regularización L2 de 5. Se compararon dos configuraciones, y solo
en la selección de 2015: la estándar, de 31 hojas y 50 ejemplos por hoja, memorizaba pedidos sueltos
y empataba con Croston. No se hizo una búsqueda más amplia de hiperparámetros, porque el margen que
importa está en las variables y en la pérdida, y una búsqueda sobre los mismos orígenes se
sobreajusta a ellos.

**ETS en semanal** va sin estacionalidad, porque la estación anual tendría 48 períodos.

**ETS corre en un proceso aparte** (`demanda._ets_proceso`). Dentro de un kernel de Jupyter en
Windows, el pool de procesos de statsforecast quedaba bloqueado sin terminar; en un proceso de Python
normal no pasa. Los datos van y vuelven en parquet y el resultado es idéntico al del llamado directo.

### 6.6 Variables (features)

Todas se calculan con datos hasta el origen `t`, nunca después.

- **Nivel** `s`: promedio de los últimos 12 meses. Si es cero, el promedio de toda la historia;
  si también es cero, 1. Las variables de cantidad y el objetivo se dividen por `s`, para que el
  modelo aprenda patrones y no tamaños; la predicción se multiplica por `s` al final.
- Rezagos `y(t), y(t−1), …, y(t−11)`. Entre ellos está `y(t−10)`, el mismo mes del año anterior.
- Promedios móviles de 3, 6 y 12 meses, desvío de 12 meses y fracción de meses sin pedidos.
- Resúmenes robustos: medianas de 6 y 12 meses, que ignoran un pedido gigante suelto, y la tasa de
  Croston SBA, que le da al modelo la mejor referencia simple como variable.
- El mes objetivo uno y dos años antes.
- `log(s)`, antigüedad de la serie en meses, mes del año del mes objetivo, depósito y categoría.
- En semanal: 8 rezagos semanales en vez de 12 mensuales, más la semana del mes y el paso.

LightGBM recibe los nulos tal cual; Ridge los imputa y agrega un indicador por variable.

**Tope del objetivo:** al entrenar, el objetivo escalado se limita a 12 veces el nivel. Un salto
mayor no se puede anticipar con estas variables y, sin tope, unas pocas series nuevas con nivel
minúsculo dominan la pérdida cuadrática de Ridge.

**Peso de cada ejemplo:** proporcional a `s`, normalizado a promedio 1. La proporción alinea la
pérdida con el WAPE; la normalización es necesaria porque, con pesos de millones de unidades, la
regularización de Ridge y la de LightGBM quedaban insignificantes. Sin normalizar, Ridge con 12
meses de historia aprendía coeficientes enormes por mes del año y su WAPE se duplicaba.

**Pérdida de LightGBM:** se elige en la selección entre L2 y L1, las dos ponderadas por `s`. Con
el peso `s`, la L1 sobre el objetivo escalado equivale al error absoluto en unidades, que es lo
que mide el WAPE. Gana la de menor WAPE; si la diferencia es menor a 1 %, gana la de menor sesgo
absoluto. Ridge usa la misma ponderación. En mensual ganó L1. Su costo es un sesgo negativo:
estima la mediana, que en series con pedidos gigantes ocasionales queda por debajo del promedio, y
`02` lo reporta. Para el escenario semanal la selección se repite en `03`, con la misma regla:
la suma de cuatro medianas semanales no es la mediana del mes y la decisión mensual no se puede
trasladar.

**Rango:** LightGBM con pérdida cuantílica en 0,1 y 0,9 da un intervalo nominal de 80 %. Su
cobertura real se mide en la prueba de 2016 y se informa aunque no llegue al 80 %.

### 6.7 Antifuga

- Las variables de un origen no pueden cambiar si se modifican los datos posteriores al origen.
  `02` lo verifica alterando el futuro y comparando.
- El entrenamiento de un origen solo usa ejemplos cuyo mes objetivo es menor o igual al origen.
- El ABC y la clase de intermitencia se calculan con toda la serie y por eso **solo segmentan
  resultados**; nunca son variables.

## 7. Notebook 01 · Limpieza y EDA

Responde: ¿qué datos tenemos, qué tan buenos son y qué ve la empresa con ellos?

1. **Preparación**: entorno, versiones y descarga.
2. **Ingesta**: lectura como texto y perfil de columnas, tipos, nulos y únicos. Muestra el tope de
   filas de Excel.
3. **Calidad ISO/IEC 25012**: una tabla con una fila por característica (completitud, exactitud,
   consistencia, actualidad, credibilidad y trazabilidad), el hallazgo y la cifra. Se exporta.
4. **Limpieza** paso a paso, cada paso con cuántas filas y cuánto volumen toca: texto a número,
   paréntesis a negativo, fechas, descarte sin fecha, recorte 2012–2016 y separación de
   devoluciones.
5. **Panel mensual** producto × depósito, con ceros desde el primer pedido.
6. **EDA**, un gráfico por pregunta, siguiendo la tabla 1.6 de la guía de estudio:
   - evolución mensual total y por depósito (línea);
   - Pareto de productos y clase ABC (barras más línea acumulada);
   - volumen por categoría (barras horizontales ordenadas);
   - estacionalidad, mes × año (mapa de calor);
   - intermitencia ADI/CV² (Syntetos y Boylan, 2005), en dispersión con los cuatro cuadrantes;
   - distribución del tamaño de pedido (escala logarítmica);
   - altas y bajas de series en el tiempo.
7. **Qué necesita la empresa para tener BI**: la línea de cierre del escalón 2, con las cifras
   del notebook.
8. **Exportación** a `data/processed/` y `resultados/powerbi/`.

## 8. Notebook 02 · Modelo de pronóstico

Responde: con todos los datos disponibles, ¿el ML le gana a los métodos simples, y por cuánto?

1. Preparación y carga del panel de `01`.
2. **El problema**: serie, horizonte `t + 2`, por qué validación temporal y no división al azar.
3. **Variables**, con la verificación antifuga.
4. **Métricas**, verificadas con el ejemplo numérico de la guía de estudio (MAE 46,7; WAPE 23,3 %;
   sesgo +10 %).
5. **Selección en 2015**: ventana de la media móvil y pérdida de LightGBM.
6. **Prueba en 2016**: los siete modelos, con tabla de promedio, desvío y peor origen, y un
   gráfico de WAPE por origen.
7. **Dónde gana y dónde pierde**: por depósito, por clase ABC y por intermitencia.
8. **Modelo final y pronóstico de febrero de 2017**, con datos hasta diciembre de 2016, exportado
   para Power BI junto con el P10, el P90 y la media móvil.
9. **Qué mira el modelo**: importancia de variables de LightGBM y coeficientes de las variables
   numéricas de Ridge, ambos del modelo final.
10. **Rango**: cuantiles 0,1 y 0,9 y su cobertura medida en 2016.
11. **Demo**: una serie con su historia, el pronóstico y el rango. Corre en segundos.
12. **Anexo · clasificación**: regresión logística para «¿habrá pedido en `t + 2`?», con la
    trampa del accuracy medida en datos reales.

Los resultados del backtest se guardan en `resultados/tablas/`. Una bandera `RECALCULAR` al inicio
permite releerlos en la demo sin volver a entrenar.

## 9. Notebook 03 · Experimento de madurez

Responde: ¿desde qué nivel de historia, calidad y frecuencia vale la pena el ML?

Cada escenario repite el backtest de 6 orígenes con todos los modelos de regresión. Se cambia un
factor por vez y los demás quedan en el escenario base: mensual, historia completa y 0 % de
faltantes.

| Factor | Niveles | Cómo se simula |
|---|---|---|
| Historización | 6, 12, 24 y 36 meses, y completa | Los datos anteriores a la ventana se borran antes de calcular variables y entrenar |
| Calidad | 0, 5, 15 y 30 % de registros perdidos | Se borran al azar filas del dataset limpio (con semilla) y se rearma el panel. Simula ventas que nunca se cargaron |
| Frecuencia | Mensual y semanal | El mes se parte en cuatro semanas (días 1–7, 8–14, 15–21 y 22 a fin de mes), se pronostican las cuatro semanas de `t + 2` y se suman |

La **evaluación es siempre contra la demanda real completa**, porque la empresa vive la demanda
real aunque haya registrado mal la suya.

**Criterio declarado antes de correr el experimento.** El ML vale la pena en un escenario cuando el
mejor modelo de ML (Ridge o LightGBM) tiene un WAPE al menos **5 % menor** que la mejor referencia
simple (naive, naive estacional, media móvil o Croston) **y** le gana en al menos 4 de los 6
orígenes. El umbral de 5 % es un supuesto de consultoría: una mejora menor no paga mantener un
pipeline en Python. ETS se reporta aparte como la alternativa estadística.

Salidas: una tabla por factor, un gráfico de WAPE contra meses de historia, uno de sesgo contra
porcentaje de faltantes y uno de mensual contra semanal. Al final, **la respuesta a la pregunta**
armada con las cifras del experimento, sin números escritos a mano. Los resultados se guardan y
una bandera `RECALCULAR` evita volver a correrlo.

## 10. Salidas para Power BI

Se exportan a `resultados/powerbi/` en CSV UTF-8, con separador coma y punto decimal.

| Archivo | Grano | Columnas |
|---|---|---|
| `dim_serie.csv` | Serie | `serie`, `producto`, `deposito`, `categoria`, `clase_abc`, `intermitencia`, `primer_mes`, `ultimo_mes` |
| `hecho_demanda_mensual.csv` | Serie × mes | `serie`, `mes`, `demanda`, `devoluciones`, `pedidos` |
| `calidad_datos.csv` | Característica | `caracteristica`, `hallazgo`, `cifra`, `filas_afectadas` |
| `hecho_pronostico_backtest.csv` | Serie × mes × modelo | `serie`, `mes`, `modelo`, `real`, `pronostico` |
| `hecho_pronostico_futuro.csv` | Serie | `serie`, `mes`, `pronostico`, `p10`, `p90`, `pronostico_media_movil` |
| `metricas_modelos.csv` | Modelo × origen | `modelo`, `origen`, `mes`, `mae`, `wape`, `sesgo`, `mase` |

El modelo de Power BI queda en estrella: `dim_serie` se relaciona con los hechos por `serie`.

## 11. Reglas de implementación

- **Ninguna cifra escrita a mano** en conclusiones: se imprimen desde las variables.
- Las figuras se guardan en `resultados/figuras/` con nombre fijo y con título, ejes y unidades en
  español.
- El notebook entregado **queda ejecutado**, con las salidas guardadas, de arriba abajo y sin
  errores.
- Cada sección cierra con una celda «Qué nos dice esto».
- El código de `src/demanda/` lleva docstrings y comentarios en español.

## 12. Terminado cuando

- [x] Los tres notebooks corren de arriba abajo, con las salidas guardadas y sin errores (en local,
      Windows, Python 3.12).
- [x] `01` reproduce los hechos de la sección 3 y escribe todas las salidas de la sección 10 que
      le tocan.
- [x] `02` pasa la verificación antifuga, reproduce el ejemplo de métricas de la guía y reporta los
      siete modelos con su dispersión.
- [x] `03` cubre los nueve escenarios y arma la respuesta con el criterio de la sección 9.
- [x] Las tablas de Power BI existen y respetan las columnas de la sección 10.
- [x] El README explica cómo correr todo en local y en Colab.
- [ ] Abrir los notebooks una vez en Colab, desde Drive, antes de la exposición. El camino de Colab
      está escrito pero no se probó.

## 13. Pendientes fuera del repo

- Actualizar la tabla 3.2 de la guía de estudio: suma Ridge y ETS, y la logística como anexo.
- Reemplazar en la guía de estudio las cifras «se confirman al correr el notebook» por las de `01`.
- Completar la sección 3.6 de la guía con la respuesta de `03`.
- Armar el `.pbix` con las tablas de `resultados/powerbi/`.
