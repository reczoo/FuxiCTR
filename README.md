# FuxiCTR

This is a fork from the official release at https://github.com/huawei-noah/benchmark/tree/main/FuxiCTR.

Click-through rate (CTR) prediction is an critical task for many industrial applications such as online advertising, recommender systems, and sponsored search. FuxiCTR builds an open-source library for CTR prediction, with stunning features in configurability, tunability, and reproducibility. It also supports the development of [Open-CTR-Benchmark](https://openbenchmark.github.io/ctr-prediction), making open benchmarking for CTR prediction available.


## Model List

| Publication| Model  | Paper | Available | 
| :-----: | :-------: |:------------|:----------:|
| WWW'07| [LR](./fuxictr/pytorch/models/LR.py)  |[Predicting Clicks: Estimating the Click-Through Rate for New Ads](https://dl.acm.org/citation.cfm?id=1242643) | :heavy_check_mark: |
|ICDM'10 | [FM](./fuxictr/pytorch/models/FM.py)  | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)| :heavy_check_mark: |
|CIKM'15| [CCPM](./fuxictr/pytorch/models/CCPM.py) | [A Convolutional Click Prediction Model](http://www.escience.cn/system/download/73676) | :heavy_check_mark: |
| RecSys'16 | [FFM](./fuxictr/pytorch/models/FFM.py) | [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134) |:heavy_check_mark: |
| RecSys'16 | [YoutubeDNN](./fuxictr/pytorch/models/DNN.py) | [Deep Neural Networks for YouTube Recommendations](http://art.yale.edu/file_columns/0001/1132/covington.pdf) |:heavy_check_mark: |
| DLRS'16 | [Wide&Deep](./fuxictr/pytorch/models/WideDeep.py)  | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) |:heavy_check_mark: |
|ECIR'16 | [FNN](./fuxictr/pytorch/models/FNN.py) | [Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) |:heavy_check_mark: |
| ICDM'16 | [IPNN](./fuxictr/pytorch/models/PNN.py) | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf) | :heavy_check_mark: |
| KDD'16 | [DeepCross](./fuxictr/pytorch/models/DeepCrossing.py) | [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)  | :heavy_check_mark: |
| NIPS'16 | [HOFM](./fuxictr/pytorch/models/HOFM.py) | [Higher-Order Factorization Machines](https://papers.nips.cc/paper/6144-higher-order-factorization-machines.pdf) | :heavy_check_mark: |
| IJCAI'17 | [DeepFM](./fuxictr/pytorch/models/DeepFM.py) | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | :heavy_check_mark: |
|SIGIR'17 | [NFM](./fuxictr/pytorch/models/NFM.py) | [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777) | :heavy_check_mark: |
|IJCAI'17 | [AFM](./fuxictr/pytorch/models/AFM.py) | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/0435.pdf) |:heavy_check_mark:|
| ADKDD'17 | [DCN](./fuxictr/pytorch/models/DCN.py)  | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | :heavy_check_mark:|
| WWW'18 | [FwFM](./fuxictr/pytorch/models/FwFM.py) | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)  | :heavy_check_mark: |
|KDD'18 | [xDeepFM](./fuxictr/pytorch/models/xDeepFM.py) | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) | :heavy_check_mark: |
|KDD'18 | [DIN](./fuxictr/pytorch/models/DIN.py) | [Deep Interest Network for Click-Through Rate Prediction](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction) | :heavy_check_mark: |
|CIKM'19 | [FiGNN](./fuxictr/pytorch/models/FiGNN.py) | [FiGNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction](https://arxiv.org/abs/1910.05552) | :heavy_check_mark: |
|CIKM'19 | [AutoInt+](./fuxictr/pytorch/models/AutoInt.py) | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) | :heavy_check_mark: |
|RecSys'19 | [FiBiNET](./fuxictr/pytorch/models/FiBiNET.py) | [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433) | :heavy_check_mark: |
|WWW'19 | [FGCNN](./fuxictr/pytorch/models/FGCNN.py) | [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/abs/1904.04447) | :heavy_check_mark: |
| AAAI'19| [HFM+](./fuxictr/pytorch/models/HFM.py) | [Holographic Factorization Machines for Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/4448)  | :heavy_check_mark: |
| Neural Networks'20 | [ONN](./fuxictr/pytorch/models/ONN.py)  | [Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579)  | :heavy_check_mark: |
| AAAI'20 | [AFN+](./fuxictr/pytorch/models/AFN.py) | [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768) | :heavy_check_mark: |
| AAAI'20  | [LorentzFM](./fuxictr/pytorch/models/LorentzFM.py) | [Learning Feature Interactions with Lorentzian Factorization](https://arxiv.org/abs/1911.09821) | :heavy_check_mark: |
| WSDM'20 | [InterHAt](./fuxictr/pytorch/models/InterHAt.py) | [Interpretable Click-through Rate Prediction through Hierarchical Attention](https://dl.acm.org/doi/10.1145/3336191.3371785) | :heavy_check_mark: |
| DLP-KDD'20 | [FLEN](./fuxictr/pytorch/models/FLEN.py) | [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/abs/1911.04690) | :heavy_check_mark: |
| WWW'21 | [FmFM](./fuxictr/pytorch/models/FmFM.py) | [FM^2: Field-matrixed Factorization Machines for Recommender Systems](https://arxiv.org/abs/2102.12994) | :heavy_check_mark: |



## Dependency
FuxiCTR has the following requirements to install. While the implementation of FuxiCTR should support more pytorch versions, we currently perform the tests on pytorch 1.0.x-1.1.x only.

+ python 3.6.x
+ pytorch 1.0.x-1.1.x
+ pandas
+ numpy
+ h5py
+ pyyaml

## Get Started

#### 1. Run the demo

Please follow [the examples](./demo/DeepFM_demo.py) in the demo directory to get started. The code workflow is structured as follows:

```python
# Set the data config and model config
feature_cols = [{...}] # define feature columns
label_col = {...} # define label column
params = {...} # set data params and model params

# Set the feature encoding specs
feature_encoder = FeatureEncoder(feature_cols, label_col, ...) # define the feature encoder
feature_encoder.fit(...) # fit and transfrom the data

# Load data generators
train_gen, valid_gen, test_gen = data_generator(feature_encoder, ...)

# Define a model
model = DeepFM(...)

# Train the model
model.fit_generator(train_gen, validation_data=valid_gen, ...)

# Evaluation
model.evaluate_generator(test_gen)

```

#### 2. Run the benchmark with given experiment_id

For reproducing the experiment result, you can run the benchmarking script with the corresponding config file as follows.

+ --config: The config directory of data and model config files.
+ --expid: The specific experiment_id that records the detailed data and model settings.
+ --gpu: The gpu index used for experiment, and -1 for CPU.

In the following example, `DeepFM_test` corresponds to an expid with specific model and dataset configurations located in [config/model_config/tests.yaml](./config/model_config/tests.yaml#L145).

```bash
cd benchmarks
python run.py --config ../config --expid DeepFM_test --gpu 0

```

#### 3. Tune the model hyper-parameters

For tuning model hyper-parameters, you can apply grid-search over the specified tuning space with the following script.

+ --config: The config file that defines the tuning space
+ --tag: (optional) Specify the tag to determine which expid to run (e.g. 001 for the first expid). This is useful to rerun one specific experiment_id that contains the tag.
+ --gpu: The available gpus for parameters tuning (e.g., setting --gpu 0 1 for two gpus)

In the following example, [FM_criteo_x4_tuner_config_01.yaml](./benchmarks/FM_criteo_x4_001/FM_criteo_x4_tuner_config_01.yaml) is a demo configuration file that defines the tuning space for parameter tuning.

```bash
cd benchmarks
python run_param_tuner.py --config ./FM_criteo_x4_001/FM_criteo_x4_tuner_config_01.yaml --gpu 0 1

```

For more running examples, please refer to the "Reproduce-Steps" of benchmarking results in [Open-CTR-Benchmark](https://openbenchmark.github.io/ctr-prediction).

## Code Structure
[Check an overview of code structure](./docs/FuxiCTR_overview.jpg) for more details on API design.


## License
The MIT License