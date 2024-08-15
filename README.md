# Proyecto Machine Learning CNN
======
En el presente se documenta el proceso de desarrollo de una red neural convolucional (CNN) para identificar neumonia en radiografías de pulmones.
### Dataset utilizado
Para el entrenamiento de la red de decidió usar el dataset público en kaggle que se encuentra en el siguiente link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  
### Preparación de los datos
* El propósito establecido para la red neuronal es clasificar entre pulmones sanos, afectados por neumonia viral, y afectados por neumonia bacteriana, por ello, el dataset fue manualmente juntado otra vez, dejando las imagenes de los pulmones en las carpetas de NORMAL y PNEUMONIA.  
* Ya teniendo las imagenes en google drive, se usan funciones de os y shutil para crear las carpetas respectivas y separar las imagenes en tres subcarpetas, normal, bacteriana y virales, para ello tomamos ventaja también de que el dataset trae las imagenes de los pulmones afectados, debidamente nombradas con el tipo de afección en el nombre de la imagen. 
#### Carga de los datos
Para la carga de datos se tenían que tener en cuenta algunas cosas, las imagenes están originalmente a una resolución media alta, aproximadamente algo superior a 1000x1000, lo que generaría demoras en el entrenamiento, también se tienen que etiquetar los datos, para ello, además de que ya están separados en carpetas, sus nombres también indican a donde pertenecen.  
1. Se generan los path de los respectivos directorios.
2. Se carga la lista de las imagenes que hay, se usa random para tener una distribución aleatoria de los mismos.
3. Se recorren los directorios calculando el número de imagenes que deben ir a train, test y validation.
4. Se corta la lista de los archivos según el número de imagenes que debe ir en cada conjunto de datos
5. Se crean las subcarpetas correspondientes y se mueven las imagenes.  
* Con las subcarpetas y las imagenes separadas respectivamente se separaron en tres subcarpetas más en cada una, datos de train, validation y test, teniendo una proporción de 70% para train, y 15% tanto para validation como test.
##### Etiquetado
El etiquetado de los datos se lleva a cabo con las carpetas creadas previamente, para ello
