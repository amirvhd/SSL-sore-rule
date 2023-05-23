
# 

This repository is for "Probabilistic Self-supervised Learning via
Scoring Rules Minimization" paper. 



## Table of contents
* [Badges](#general-information)
* [Installation](#Installation)
* [Usage/Examples](#Usage/Examples)
* [Acknowledgements](#Acknowledgements)

## Badges

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)

### Dependency

![Python](https://img.shields.io/badge/Python-3.9-brightgreen)
![torch](https://img.shields.io/badge/Torch-1.10.1-brightgreen)

## Installation

Install requirments:
```python
pip install -r requirements.txt
```


## Usage/Examples


### Pretraining

You can run the pretraining of model for Imagenet with following code.

```python
torchrun --nproc_per_node=8 main_prosmin.py --arch vit_small 
``` 
### Linear evaluation

You can run the linear-evaluation of the model for Imagenet with the following code. 

```python
torchrun --nproc_per_node=8 eval_linear.py --arch vit_small
``` 

### Uncertainty evaluation 

```python
python 
```
 
### Out of distribtuion detection


```python
python 
```



## Acknowledgements
Base DINO adapted from following repository:

 - [DINO](https://github.com/facebookresearch/dino)
