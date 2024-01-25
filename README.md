# KnowledgeTransferGraph

This repository contains the implementation of the "Knowledge Transfer Graph for Deep Collaborative Learning" as described in the [arXiv paper](https://arxiv.org/abs/1909.04286). In this implementation, the original CorrectGate has been replaced by NegativeLinearGate.

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
