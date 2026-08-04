# ChronosAD: Leveraging Time Series Foundation Models for Accurate Anomaly Detection

The official implementation of the paper [ChronosAD: Leveraging Time Series Foundation Models for Accurate Anomaly Detection](https://arxiv.org/abs/2606.01300v1) accepted at the 24th IEEE International Conference on Industrial Informatics (INDIN 2026).

## Abstract
Time series anomaly detection is a crucial task in various domains, including finance, healthcare, and industry. However, existing methods often struggle to generalize across different datasets, especially when anomalies are subtle or context-dependent. To solve this issue, we introduce ChronosAD, a novel architecture for anomaly detection that uses a time series foundation model as a feature extractor. Specifically, it employs a two-stage pipeline: first, it uses the foundation model to extract embeddings for each time series in a zero-shot manner. Then, a custom-developed Temporal Block, composed of Bidirectional Long Short-Term Memory (BiLSTM) and Multi-Head Attention, refines these embeddings to capture temporal dependencies and highlight salient patterns. Unlike previous approaches, our model requires minimal task-specific tuning and demonstrates robust generalization across a wide range of domains, including industrial, medical, cyber-physical, and automotive systems. Extensive experiments on 11 benchmarks show that ChronosAD outperforms existing methods by 4.72% in AUC and 6.60% in AP on average.

## Installation ##

**1. Repository setup:**
```
$ git clone https://github.com/intelligolabs/ChronosAD.git
$ cd ChronosAD
```
**2. Environment setup:**
```
$ conda create -n ChronosAD python=3.10
$ conda activate ChronosAD
$ pip install git+https://github.com/amazon-science/chronos-forecasting.git
$ pip install -r requirements.txt
```
**3. Training and Testing:**
```
$ python main.py <dataset_path> --model ChronosLSTM --dataset_name <dataset_name> --batch_size 32 --context 250 --num_epochs 30 --learning_rate 0.001 --gpu_idx 0 --qualitative_results
```

## Acknowledgement
We gratefully acknowledge the open-source contributions of the [Chronos Foundation Model](https://github.com/amazon-science/chronos-forecasting) team, whose work made this project possible.

## Authors
Uzair Khan<sup>1</sup>, Luigi Capogrosso<sup>2</sup>, Francesco Biondani<sup>1</sup>, Michele Magno<sup>3,2</sup>, Franco Fummi<sup>1</sup>, Francesco Setti<sup>1</sup>, Marco Cristani<sup>1,4</sup>

<sup>1</sup> *University of Verona*
<sup>2</sup> *Interdisciplinary Transformation University of Austria*
<sup>3</sup> *ETH Zurich*
<sup>4</sup> *Reykjavik University*