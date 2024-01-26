# KnowledgeTransferGraph

This repository implements the "Knowledge Transfer Graph for Deep Collaborative Learning" described in the [arXiv paper](https://arxiv.org/abs/1909.04286). Notably, this implementation deviates from the original paper in certain aspects related to hyperparameter tuning.

## Changes Made to Hyperparameter Tuning

In the original paper, the authors employed ASHA for hyperparameter tuning with pruning. However, it was deemed unsuitable for optimizing the Gate function in the temporal dimension. Consequently, pruning using ASHA was disabled, and a NopPruner (no pruning) was adopted as the pruner.

Moreover, the original CorrectGate was replaced by NegativeLinearGate in the implementation. This change was made to specifically enhance the model's performance in controlling knowledge transfer along the temporal dimension, with the aim of improving overall accuracy.

Additionally, the original RandomSampler for sampling was replaced with a Multivariate TPE (Tree-structured Parzen Estimator) sampler to enhance the efficiency of the search process.

## Usage
To use the Knowledge Transfer Graph in your project, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/yukiharada1228/KnowledgeTransferGraph.git
cd KnowledgeTransferGraph
```
2. Install the package:
```bash
pip install .
```

## Replace CorrectGate with NegativeLinearGate
In the source code, locate the implementation of the gate function. In the original paper, CorrectGate is used. In this implementation, it has been replaced by NegativeLinearGate. 

## Acknowledgements

This implementation is based on the original paper ["Knowledge Transfer Graph for Deep Collaborative Learning"](https://arxiv.org/abs/1909.04286) by Soma Minami, Tsubasa Hirakawa, Takayoshi Yamashita, and Hironobu Fujiyoshi. I acknowledge and appreciate their valuable contributions to the field.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
