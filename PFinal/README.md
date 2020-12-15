# Proyecto Final Aprendizaje de Máqina

## Descripción

Proyecto elaborado para la materia de Aprendizaje de Máquina impoartida en el ITAM durante el semestre Otoño 2020. Propusimos usar GANs (Generative Adversarial Networks)
y entrenar una red que conviertiera imágenes normales a pinturas al estilo de Vincent van Gogh. Para lograr esto; primero debimos entender qué es tensorflow y aprender
a usarlo, luego entender la historia de transferencia de estilo en imágenes, después aprender a manejar GANs y finalmente usarlas para nuestro proyecto.

El proyecto está basado en el paper [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593), si se desea
replicar el proyecto también recomendamos leer los siguientes papers: [Primero](https://arxiv.org/abs/1502.03167) y [Segundo](https://arxiv.org/abs/1607.08022). 

## Set-Up
Todo el código del proyecto fue corriedo en una máquina con procesador Intel Core i7-9750H @2.60GHz y una tarjeta de video nvidia GTX 1660 Ti en su versión para 
laptop. La configuración importante del ambiente de programación es la siguiente:
* python 3.x con jupyter notebooks
* tensorflow-gpu 2.3.1
* CUDA 10.1
* Pop!_OS 20.10


En caso de tener versiones posteriores de cuda recomendamos [este](https://medium.com/@praveenkrishna/downgrade-cuda-for-tensorflow-gpu-17831db59099) post que explica
cómo hacer un downgrade de los drivers. Si se va a realizar un fresh install el siguiente 
[post](https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d) es bueno.

Para revisar tensorflow encontró un GPU ejecutar el siguiente [código](https://github.com/FranciscoBuru/ML/blob/master/PFinal/src/Untitled.ipynb), la salida debe
coincidir con la que se muestra.

## Transferencia de estilos

## Introducción a GANs

## GANs para transferencia de estilo

## Proyecto van Gogh
### Set-Up
### Entrenamiento
### Resultados
