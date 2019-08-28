# ImageClassifier-Udacity-Machine-Learning-Nanodegree-Project

Project implementation of for Udacity's Introduction to Machine Learning Nanodegree program. In this project, students first develop an image classifier built with PyTorch then convert it into a command prompt application.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

## Image Classifier Project 

Project Files:

- `train.py:` this invoves training a new model on the datasets.
- `predict.py:` this involves predicting the flower name from the image datasets.
- `Image Classifier Project.ipynb:` Project implementation on Jupyter Notebook 
- `Image Classifier Project.html:` HTML export of the Jupyter Notebook of the above project implementation.


## Datasets

The datasets image categories is found in  [cat to name.json] and flower images can be downloaded in the gziped tar file [flower_data.tar.gz] from Udacity.


The dataset Image categories is found in [cat_to_name.json] and flower images can be downloaded from [flower_data.tar.gz] from Udacity.

You can also download the flower images:

```bash
   https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz | tar xz
```
## Poject Implementation on Jupyter Notebook

```Image Classifier Project.ipynb```

 Launch **Jupyter Notebook** from the project root to review  ```Image Classifier Project.ipynb``` notebook file


## Example of Command line train.py

Help

```
python ./train.py -h

```

Training on **CPU** with  **vgg16** you can use:
```
python ./train.py ./flowers/train/
```

Training  on **GPU** 
```
python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_layers 3136 --epochs 5
```


## Example of Command line predict.py 

Help

```
python ./predict.py -h
```

Make Prediction with gpu
```
python ./predict.py flowers/valid/10/image_07101.jpg my_checkpoint.pth --gpu
```
