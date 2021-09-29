# FuxiCTR

Click-through rate (CTR) prediction is an critical task for many industrial applications such as online advertising, recommender systems, and sponsored search. FuxiCTR provides an open-source library for CTR prediction, with stunning features in configurability, tunability, and reproducibility. 


## Model List

| Publication| Model  | Paper | Available | 
| :-----: | :-------: |:------------|:----------:|
| WWW'07| LR  |[Predicting Clicks: Estimating the Click-Through Rate for New Ads](https://dl.acm.org/citation.cfm?id=1242643) | :heavy_check_mark: |
|ICDM'10 | FM  | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)| :heavy_check_mark: |
|CIKM'15| CCPM | [A Convolutional Click Prediction Model](http://www.escience.cn/system/download/73676) | :heavy_check_mark: |
| RecSys'16 | FFM | [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134) |:heavy_check_mark: |
| RecSys'16 | YoutubeDNN | [Deep Neural Networks for YouTube Recommendations](http://art.yale.edu/file_columns/0001/1132/covington.pdf) |:heavy_check_mark: |
| DLRS'16 | Wide&Deep  | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) |:heavy_check_mark: |
|ECIR'16 | FNN | [Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) |:heavy_check_mark: |
| ICDM'16 | IPNN | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf) | :heavy_check_mark: |
| KDD'16 | DeepCross | [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)  | :heavy_check_mark: |
| NIPS'16 | HOFM | [Higher-Order Factorization Machines](https://papers.nips.cc/paper/6144-higher-order-factorization-machines.pdf) | :heavy_check_mark: |
| IJCAI'17 | DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | :heavy_check_mark: |
|SIGIR'17 | NFM | [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777) | :heavy_check_mark: |
|IJCAI'17 | AFM | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/0435.pdf) |:heavy_check_mark:|
| ADKDD'17 | DCN  | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | :heavy_check_mark:|
| WWW'18 | FwFM | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)  | :heavy_check_mark: |
|KDD'18 | xDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) | :heavy_check_mark: |
|KDD'18 | DIN | [Deep Interest Network for Click-Through Rate Prediction](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction) | :heavy_check_mark: |
|CIKM'19 | FiGNN | [FiGNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction](https://arxiv.org/abs/1910.05552) | :heavy_check_mark: |
|CIKM'19 | AutoInt+ | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) | :heavy_check_mark: |
|RecSys'19 | FiBiNET | [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433) | :heavy_check_mark: |
|WWW'19 | FGCNN | [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/abs/1904.04447) | :heavy_check_mark: |
| AAAI'19| HFM+ | [Holographic Factorization Machines for Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/4448)  | :heavy_check_mark: |
| Neural Networks'20 | ONN  | [Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579)  | :heavy_check_mark: |
| AAAI'20 | AFN+ | [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768) | :heavy_check_mark: |
| AAAI'20  | LorentzFM | [Learning Feature Interactions with Lorentzian Factorization](https://arxiv.org/abs/1911.09821) | :heavy_check_mark: |
| WSDM'20 | InterHAt | [Interpretable Click-through Rate Prediction through Hierarchical Attention](https://dl.acm.org/doi/10.1145/3336191.3371785) | :heavy_check_mark: |
| DLP-KDD'20 | FLEN | [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/abs/1911.04690) | :heavy_check_mark: |


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

#### 2. Run the benchmark with given experiment_ID

For reproducing the experiment result, you can run the benchmarking script with the corresponding config file as follows.

+ --config: The config directory of data and model config files.
+ --expid: The specific experiment_ID that records the detailed data and model settings.
+ --gpu: The gpu index used for experiment, and -1 for CPU.

For example, `DeepFM_test` is an expid located in `config/model_config/tests.yaml`.

```bash
cd benchmarks
python benchmark.py --config ../config --expid DeepFM_test --gpu 0

```

#### 3. Tune the model hyper-parameters

For model hyper-parameters tuning, you can apply grid-search over the specified tuning space with the following script.

+ --config: The config file that defines the tuning space
+ --gpu: The available gpus for parameters tuning.

```bash
cd benchmarks
python run_param_tuner.py --config ./FM_criteo_x4_001/FM_criteo_x4_tuner_config_01.yaml --gpu 0

```

## Code Structure
[Check an overview of code structure](./docs/FuxiCTR_overview.jpg) for more details on API design.


## License
The MIT License
