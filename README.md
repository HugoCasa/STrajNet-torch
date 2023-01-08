# PyTorch implementation of STrajNet

This repo is a pytorch implementation of the STrajNet model.

## Description of changes

- Updated preprocessing to generate compressed numpy files for each scenario ([link to file in other repo](https://github.com/HugoCasa/STrajNet/blob/master/numpy_preprocessing.py))
- Implemented distributed training with validation using DDP
- Implemented inference to evaluate on waymo website

## Original STrajNet paper

**STrajNet: Multi-Model Hierarchical Transformer for Occupancy Flow Field Prediction in Autonomous Driving**
<br> [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Zhiyu Huang](https://mczhi.github.io/), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[arXiv]](http://arxiv.org/abs/2208.00394)**&nbsp;
