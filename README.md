
# ProSMin: Probabilistic Self-supervised Learning via Scoring Rules Minimization

This repository is for!["Probabilistic Self-supervised Learning via
Scoring Rules Minimization"](https://epub.ub.uni-muenchen.de/121741/1/5704_Probabilistic_Self_superv.pdf)  paper. 



<img src="Framework.png" width="600" alt="framework">



## Table of contents
* [Installation](#Installation)
* [Dataset](#Dataset)
* [Usage/Examples](#Usage/Examples)
* [Acknowledgements](#Acknowledgements)
* [License](#License)

### Dependency

![Python](https://img.shields.io/badge/Python-3.9-brightgreen)
![torch](https://img.shields.io/badge/Torch-1.12.1-brightgreen)

## Installation

Install requirments:
```python
pip install -r requirements.txt
```
## Dataset

Please download [Imagenet](https://image-net.org/). 

## Usage/Examples


### Pretraining

You can run the pretraining of model for Imagenet with following code.

```python
torchrun --nproc_per_node=8 main_prosmin.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
``` 
### Linear evaluation

You can run the linear-evaluation of the model for Imagenet with the following code. 

```python
torchrun --nproc_per_node=8 eval_linear.py --arch vit_small --data_path /path/to/imagenet --pretrained_weights /path/to/saving_dir
``` 

## Acknowledgement
This repository is built using the [DINO](https://github.com/facebookresearch/dino) repository.

## License
This repository is released under the Apache 2.0 license as found in the LICENSE file.



