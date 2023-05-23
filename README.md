
# ProSMin: Probabilistic Self-supervised Learning via Scoring Rules Minimization

This repository is for "Probabilistic Self-supervised Learning via
Scoring Rules Minimization" paper. 

![framework](https://github.com/amirvhd/SSL-sore-rule/assets/65691404/70be2d64-ad8b-4b5c-bbe8-c9605009e5b0)





## Table of contents
* [Installation](#Installation)
* [Usage/Examples](#Usage/Examples)
* [Acknowledgements](#Acknowledgements)
* [License](#License)

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

## Acknowledgement
This repository is built using the [DINO](https://github.com/facebookresearch/dino) repository.

## License
This repository is released under the Apache 2.0 license as found in the LICENSE file.



