# Proyecto Machine Learning CNN
## Introducción
En este proyecto se desarrolla una Red Neuronal Convolucional (CNN) con el objetivo de identificar neumonía en radiografías de pulmones. El sistema está diseñado para clasificar las imágenes de pulmones en tres categorías: pulmones sanos, pulmones afectados por neumonía viral y pulmones afectados por neumonía bacteriana. Este documento detalla el proceso seguido desde la selección y preparación del dataset hasta la implementación y el entrenamiento del modelo.
## Dataset utilizado
Para el desarrollo y entrenamiento de la CNN, se utilizó el dataset público disponible en Kaggle, accesible a través del siguiente enlace:: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. Este dataset incluye imágenes clasificadas en dos categorías principales: NORMAL (pulmones sanos) y PNEUMONIA (pulmones afectados por neumonía). Para los fines de este proyecto, se realizó una clasificación adicional para diferenciar entre neumonía viral y bacteriana.
## Preparación de los datos
### Reorganización del Dataset
El primer paso consistió en reorganizar el dataset para que se ajustara a los objetivos de clasificación de la CNN. Se crearon tres subcarpetas principales para almacenar las imágenes de:* Ya teniendo las imagenes en google drive, se usan funciones de os y shutil para crear las carpetas respectivas y separar las imagenes en tres subcarpetas, normal, bacteriana y virales, para ello tomamos ventaja también de que el dataset trae las imagenes de los pulmones afectados, debidamente nombradas con el tipo de afección en el nombre de la imagen. 
1. Pulmones Normales ('NORMAL').
2. Pulmones con Neumonía Viral ('VIRAL').
3. Pulmones con Neumonía Bacteriana ('BACTERIAL').

La clasificación de las imágenes se realizó manualmente, aprovechando que los nombres de los archivos en el dataset original ya indican el tipo de afección.
## Carga y Preprocesamiento de los Datos
El proceso de carga y preprocesamiento de los datos consideró varios factores para asegurar un entrenamiento eficiente del modelo:

1. **Reducción de Resolución**: Las imágenes originales tienen una resolución alta (superior a 1000x1000 píxeles), lo cual podría generar demoras significativas durante el entrenamiento. Para mitigar este problema, se optó por reducir la resolución de las imágenes a un tamaño más manejable, preservando al mismo tiempo la calidad suficiente para la clasificación.

2. **Etiquetado Automático**: Dado que las imágenes ya están organizadas en carpetas y sus nombres indican la categoría correspondiente, se utilizó un proceso automatizado para generar las etiquetas necesarias para el modelo.
   
3. **Distribución de Datos**: Se implementó un procedimiento para dividir las imágenes en conjuntos de entrenamiento, validación y prueba. Las imágenes fueron distribuidas en una proporción de 70% para el entrenamiento, y 15% tanto para la validación como para la prueba. Para asegurar que las imágenes en cada conjunto estén distribuidas aleatoriamente, se utilizó la función random para barajar la lista de imágenes antes de dividirlas.

4. **Creación de Subcarpetas**: Se crearon subcarpetas dentro de cada categoría principal (NORMAL, VIRAL, BACTERIAL) para separar las imágenes en los tres conjuntos mencionados (train, validation, test). Las imágenes fueron movidas a estas subcarpetas de acuerdo con la distribución establecida.
## Entrenamiento del Modelo
El siguiente paso en el proyecto es el entrenamiento del modelo de CNN utilizando los datos preprocesados. Este proceso incluirá la definición de la arquitectura de la red, la selección de hiperparámetros adecuados, y la evaluación del modelo utilizando los conjuntos de validación y prueba.

Para el entrenamiento, se considerarán distintas arquitecturas de CNN que se ajusten a la tarea de clasificación de imágenes, prestando especial atención a la profundidad de la red y la cantidad de filtros utilizados en cada capa convolucional. Se planea experimentar con varias configuraciones para encontrar la que ofrezca el mejor rendimiento en términos de precisión y generalización.

### 1. Ajuste Inicial de la Arquitectura:

- Arquitectura Simple: Inicialmente, se utilizó una arquitectura sencilla con pocas capas y un kernel de tamaño 5x5. Sin embargo, este diseño no arrojó buenos resultados en cuanto a precisión, lo que motivó a realizar ajustes en la arquitectura.
  
- Cambios en la Arquitectura: Se aumentó el número de capas y se cambió el kernel a un tamaño 3x3, que es más estándar en arquitecturas de redes neuronales convolucionales. Además, se probó el optimizador Adam en lugar de Stochastic Gradient Descent (SGD) para mejorar el estancamiento en la precisión.

### 2. Optimización y Overfitting:

- Resultados con Adam: El optimizador Adam mostró una mejora significativa en la precisión del modelo, alcanzando una precisión de 1 en el conjunto de entrenamiento. Sin embargo, la precisión en el conjunto de validación se estancó, lo que indicaba un claro problema de sobreajuste.
  
- Uso de SGD y Early Stopping: Para abordar el sobreajuste, se decidió volver a utilizar SGD como optimizador y entrenar con una técnica de early stopping, que detiene el entrenamiento cuando la precisión de validación deja de mejorar.

### 3. Técnicas de Data Augmentation:

- Reincorporación de Adam: Se observó que Adam continuaba proporcionando mejores resultados y de manera más rápida que SGD, por lo que se decidió reincorporarlo como optimizador. Para mitigar el sobreajuste, se aplicaron técnicas de data augmentation.
  
- Dropout y Ajustes de Data Augmentation: Aunque el data augmentation ayudó a reducir el sobreajuste, la precisión en la validación no mejoró significativamente. Se decidió entonces reducir los parámetros de data augmentation y agregar una capa de dropout por defecto para mejorar la generalización del modelo.

### 4. Evaluación de Métricas y Ajustes Finales:

- Experimentos con Data Augmentation y Dropout: Se comprobó que reducir el data augmentation y utilizar dropout producía mejores resultados. Posteriormente, se decidió eliminar el data augmentation y dejar únicamente el dropout, enfocándose en la métrica de recall, dado que en el contexto médico es crucial minimizar los falsos negativos.
  
- Modificación de Capas y Filtros: Se realizaron varios experimentos ajustando el número de filtros y capas. Inicialmente, se redujeron los filtros a la mitad y se agregó una nueva capa de 64 filtros. Luego, se incrementó a 128 filtros, pero al observar un deterioro en las métricas, se optó por mantener un máximo de 64 filtros. Finalmente, se decidió poner todas las capas a 64 filtros y aumentar el tamaño del kernel, reduciendo una capa de max pooling para mejorar los resultados.
  
- Arquitectura Final: Con base en estos experimentos, se decidió adoptar la arquitectura que mostró el mejor desempeño, analizando los resultados con la nueva métrica de recall.

## Justificacion de herramientas

- **Optimizador Adam**: Se utilizó por su eficiencia y efectividad en el ajuste de la tasa de aprendizaje de manera adaptativa.

- **Función de Pérdida**: La entropía cruzada categórica se empleó debido a la naturaleza de clasificación multiclase del problema.

- **Métricas de Evaluación**: Precisión y recall fueron monitorizadas para evaluar el rendimiento del modelo tanto en la identificación correcta de las clases como en la minimización de falsos negativos, crucial en aplicaciones médicas.

- **Early Stopping**: Se configuró para detener el entrenamiento si la precisión de validación no mejora tras varias épocas, ajustando el modelo para obtener los mejores pesos posibles sin inclinarse al sobreajuste.

- **Precisión (Accuracy)**: Refleja la capacidad general del modelo para clasificar correctamente las imágenes en todas las categorías. A lo largo de las pruebas, se observaron variaciones significativas en la precisión entre el conjunto de entrenamiento y de validación, lo que inicialmente sugirió la posibilidad de sobreajuste. La precisión alcanzó picos de perfección en el entrenamiento, lo que no se replicó en las pruebas, indicando que el modelo estaba memorizando los datos de entrenamiento en lugar de aprender características generalizables.

- **Recall**: Esta métrica es de particular importancia en aplicaciones médicas debido a las consecuencias potencialmente graves de los falsos negativos. El recall experimentó mejoras significativas a medida que se ajustaba el modelo, especialmente después de la implementación de técnicas de regularización como el dropout, que ayudaron a mejorar la sensibilidad del modelo ante las clases minoritarias.

## Implicaciones de los Resultados

La variabilidad en precisión y recall a lo largo de las pruebas enfatiza la importancia de equilibrar complejidad del modelo y capacidad de generalización. Los experimentos demostraron que una arquitectura más compleja no necesariamente se traduce en mejor rendimiento, especialmente en un contexto donde el equilibrio entre sensibilidad y especificidad es crucial.

## Áreas para Mejoras Futuras:

- **Exploración de Arquitecturas Más Profundas y Complejas**: Dado que ajustes incrementales en la profundidad y los filtros ofrecieron mejoras, explorar arquitecturas más profundas podría permitir al modelo capturar características más sutiles y complejas de las imágenes.

- **Expansión del Conjunto de Datos**: Integrar más datos, posiblemente de múltiples fuentes, podría ayudar a mejorar la robustez del modelo. Además, la inclusión de datos anotados por expertos de diversas geografías y demografías puede enriquecer el conjunto de entrenamiento y validación.

- **Innovación en Técnicas de Regularización**: Experimentar con nuevas técnicas de regularización más allá del dropout, como la normalización por lotes o capas de normalización de pesos, podría ofrecer nuevas vías para controlar el sobreajuste y mejorar la generalización.
