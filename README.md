# PyTorch implementation of STrajNet and other models

This repo includes a pytorch implementation of the STrajNet model. It also includes other models based on STrajNet as well as on FlexiViT and Perceiver IO.

## STrajnet-torch

- Updated preprocessing to generate compressed numpy files for each scenario ([link to file in other repo](https://github.com/HugoCasa/STrajNet/blob/master/numpy_preprocessing.py))
- Implemented distributed training with validation using DDP
- Implemented inference to evaluate on waymo website

## FlexiPerceiver

STrajNet, FlexiViT and Perceiver IO inspired architecture. Can be found in `flexi_perceiver.py`.

## Sources

```
@misc{https://doi.org/10.48550/arxiv.2208.00394,
  doi = {10.48550/ARXIV.2208.00394},
  url = {https://arxiv.org/abs/2208.00394},
  author = {Liu, Haochen and Huang, Zhiyu and Lv, Chen},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {STrajNet: Multi-modal Hierarchical Transformer for Occupancy Flow Field Prediction in Autonomous Driving},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

```
@misc{beyer2023flexivit,
      title={FlexiViT: One Model for All Patch Sizes}, 
      author={Lucas Beyer and Pavel Izmailov and Alexander Kolesnikov and Mathilde Caron and Simon Kornblith and Xiaohua Zhai and Matthias Minderer and Michael Tschannen and Ibrahim Alabdulmohsin and Filip Pavetic},
      year={2023},
      eprint={2212.08013},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@misc{jaegle2022perceiver,
      title={Perceiver IO: A General Architecture for Structured Inputs & Outputs}, 
      author={Andrew Jaegle and Sebastian Borgeaud and Jean-Baptiste Alayrac and Carl Doersch and Catalin Ionescu and David Ding and Skanda Koppula and Daniel Zoran and Andrew Brock and Evan Shelhamer and Olivier Hénaff and Matthew M. Botvinick and Andrew Zisserman and Oriol Vinyals and Joāo Carreira},
      year={2022},
      eprint={2107.14795},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
