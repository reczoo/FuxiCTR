<div align="center">
<img src="https://cdn.jsdelivr.net/gh/reczoo/FuxiCTR@main/docs/img/logo.png" alt="Logo" width="260"/>
</div>

<div align="center">
<a href="https://pypi.org/project/fuxictr"><img src="https://img.shields.io/badge/python-3.9+-blue" style="max-width: 100%;" alt="Python version"></a>
<a href="https://pypi.org/project/fuxictr"><img src="https://img.shields.io/badge/pytorch-1.10+-blue" style="max-width: 100%;" alt="Pytorch version"></a>
<a href="https://pypi.org/project/fuxictr"><img src="https://img.shields.io/badge/tensorflow-2.1+-blue" style="max-width: 100%;" alt="Pytorch version"></a>
<a href="https://pypi.org/project/fuxictr"><img src="https://img.shields.io/pypi/v/fuxictr.svg" style="max-width: 100%;" alt="Pypi version"></a>
<a href="https://pepy.tech/project/fuxictr"><img src="https://static.pepy.tech/badge/fuxictr" style="max-width: 100%;" alt="Downloads"></a>
<a href="https://github.com/reczoo/FuxiCTR/blob/main/LICENSE"><img src="https://img.shields.io/github/license/reczoo/fuxictr.svg" style="max-width: 100%;" alt="License"></a>
</div>
<hr/>

<div align="center">
<a href="https://github.com/reczoo/FuxiCTR/stargazers"><img src="http://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=reczoo&repo=FuxiCTR" width="600"/><a/>
</div>

Click-through rate (CTR) prediction is a critical task for various industrial applications such as online advertising, recommender systems, and sponsored search. FuxiCTR provides an open-source library for CTR prediction, with key features in configurability, tunability, and reproducibility. We hope this project could promote reproducible research and benefit both researchers and practitioners in this field.

## Key Features

+ **Configurable**: Both data preprocessing and models are modularized and configurable.

+ **Tunable**: Models can be automatically tuned through easy configurations.

+ **Reproducible**: All the benchmarks can be easily reproduced.

+ **Extensible**: It can be easily extended to any new models, supporting both Pytorch and Tensorflow frameworks.


## Model Zoo

| No  | Publication       | Model                                    | Paper                                                                                                                                                                                                           | Benchmark                                                                                                       | Version       |
|:---:|:-----------------:|:----------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------:|:-------------:|
|<tr><th colspan=6 align="center">:open_file_folder: **Feature Interaction Models**</th></tr>|
| 1   | WWW'07            | [LR](./model_zoo/LR)                     | [Predicting Clicks: Estimating the Click-Through Rate for New Ads](https://dl.acm.org/citation.cfm?id=1242643) :triangular_flag_on_post:**Microsoft**                                                           | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/LR)           | `torch`       |
| 2   | ICDM'10           | [FM](./model_zoo/FM)                     | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)                                                                                                                            | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FM)           | `torch`       |
| 3   | CIKM'13           | [DSSM](./model_zoo/DSSM)                 | [Learning Deep Structured Semantic Models  for Web Search using Clickthrough Data ](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) :triangular_flag_on_post:**Microsoft**                   | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DSSM)         | `torch`       |
| 4   | CIKM'15           | [CCPM](./model_zoo/CCPM)                 | [A Convolutional Click Prediction Model](http://www.escience.cn/system/download/73676)                                                                                                                          | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/CCPM)         | `torch`       |
| 5   | RecSys'16         | [FFM](./model_zoo/FFM)                   | [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134) :triangular_flag_on_post:**Criteo**                                                                         | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FFM)          | `torch`       |
| 6   | RecSys'16         | [DNN](./model_zoo/DNN)            | [Deep Neural Networks for YouTube Recommendations](http://art.yale.edu/file_columns/0001/1132/covington.pdf) :triangular_flag_on_post:**Google**                                                                | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DNN)          | `torch`, `tf` |
| 7   | DLRS'16           | [Wide&Deep](./model_zoo/WideDeep)        | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) :triangular_flag_on_post:**Google**                                                                                        | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/WideDeep)     | `torch`, `tf` |
| 8   | ICDM'16           | [PNN](./model_zoo/PNN)                  | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                                                                              | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/PNN)         | `torch`       |
| 9   | KDD'16            | [DeepCrossing](./model_zoo/DeepCrossing) | [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf) :triangular_flag_on_post:**Microsoft**                          | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DeepCrossing) | `torch`       |
| 10  | NIPS'16           | [HOFM](./model_zoo/HOFM)                 | [Higher-Order Factorization Machines](https://papers.nips.cc/paper/6144-higher-order-factorization-machines.pdf)                                                                                                | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/HOFM)         | `torch`       |
| 11  | IJCAI'17          | [DeepFM](./model_zoo/DeepFM)             | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) :triangular_flag_on_post:**Huawei**                                                                 | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DeepFM)       | `torch`, `tf` |
| 12  | SIGIR'17          | [NFM](./model_zoo/NFM)                   | [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777)                                                                                                     | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/NFM)          | `torch`       |
| 13  | IJCAI'17          | [AFM](./model_zoo/AFM)                   | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/0435.pdf)                                                        | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/AFM)          | `torch`       |
| 14  | ADKDD'17          | [DCN](./model_zoo/DCN)                   | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) :triangular_flag_on_post:**Google**                                                                                           | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DCN)          | `torch`, `tf` |
| 15  | WWW'18            | [FwFM](./model_zoo/FwFM)                 | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf) :triangular_flag_on_post:**Oath, TouchPal, LinkedIn, Alibaba**           | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FwFM)         | `torch`       |
| 16  | KDD'18            | [xDeepFM](./model_zoo/xDeepFM)           | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) :triangular_flag_on_post:**Microsoft**                                            | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/xDeepFM)      | `torch`       |
| 17  | CIKM'19           | [FiGNN](./model_zoo/FiGNN)               | [FiGNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction](https://arxiv.org/abs/1910.05552)                                                                                           | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FiGNN)        | `torch`       |
| 18  | CIKM'19           | [AutoInt/AutoInt+](./model_zoo/AutoInt)  | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                                                                                          | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/AutoInt)      | `torch`       |
| 19  | RecSys'19         | [FiBiNET](./model_zoo/FiBiNET)           | [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433) :triangular_flag_on_post:**Sina Weibo**                            | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FiBiNET)      | `torch`       |
| 20  | WWW'19            | [FGCNN](./model_zoo/FGCNN)               | [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/abs/1904.04447) :triangular_flag_on_post:**Huawei**                                                    | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FGCNN)        | `torch`       |
| 21  | AAAI'19           | [HFM/HFM+](./model_zoo/HFM)              | [Holographic Factorization Machines for Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/4448)                                                                                                 | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/HFM)          | `torch`       |
| 22  | Arxiv'19          | [DLRM](./model_zoo/DLRM)                 | [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091) :triangular_flag_on_post:**Facebook**                                                     | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DLRM)         | `torch`       |
| 23  | NeuralNetworks'20 | [ONN](./model_zoo/ONN)                   | [Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579)                                                                                                                | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/ONN)          | `torch`, `tf`      |
| 24  | AAAI'20           | [AFN/AFN+](./model_zoo/AFN)              | [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768)                                                                           | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/AFN)          | `torch`       |
| 25  | AAAI'20           | [LorentzFM](./model_zoo/LorentzFM)       | [Learning Feature Interactions with Lorentzian Factorization](https://arxiv.org/abs/1911.09821) :triangular_flag_on_post:**eBay**                                                                               | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/LorentzFM)    | `torch`       |
| 26  | WSDM'20           | [InterHAt](./model_zoo/InterHAt)         | [Interpretable Click-through Rate Prediction through Hierarchical Attention](https://dl.acm.org/doi/10.1145/3336191.3371785) :triangular_flag_on_post:**NEC Labs, Google**                                      | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/InterHAt)     | `torch`       |
| 27  | DLP-KDD'20        | [FLEN](./model_zoo/FLEN)                 | [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/abs/1911.04690) :triangular_flag_on_post:**Tencent**                                                                                     | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FLEN)         | `torch`       |
| 28  | CIKM'20           | [DeepIM](./model_zoo/DeepIM)             | [Deep Interaction Machine: A Simple but Effective Model for High-order Feature Interactions](https://dl.acm.org/doi/abs/10.1145/3340531.3412077) :triangular_flag_on_post:**Alibaba, RealAI**                   | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DeepIM)       | `torch`       |
| 29  | WWW'21            | [FmFM](./model_zoo/FmFM)                 | [FM^2: Field-matrixed Factorization Machines for Recommender Systems](https://arxiv.org/abs/2102.12994) :triangular_flag_on_post:**Yahoo**                                                                      | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FmFM)         | `torch`       |
| 30  | WWW'21            | [DCN-V2](./model_zoo/DCNv2)              | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) :triangular_flag_on_post:**Google**                                      | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DCNv2)        | `torch`       |
| 31  | CIKM'21           | [DESTINE](./model_zoo/DESTINE)           | [Disentangled Self-Attentive Neural Networks for Click-Through Rate Prediction](https://arxiv.org/abs/2101.03654) :triangular_flag_on_post:**Alibaba**                                                          | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DESTINE)      | `torch`       |
| 32  | CIKM'21           | [EDCN](./model_zoo/EDCN)                 | [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf) :triangular_flag_on_post:**Huawei** | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/EDCN)         | `torch`       |
| 33  | DLP-KDD'21        | [MaskNet](./model_zoo/MaskNet)           | [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619) :triangular_flag_on_post:**Sina Weibo**                                      | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/MaskNet)      | `torch`       |
| 34  | SIGIR'21          | [SAM](./model_zoo/SAM)                   | [Looking at CTR Prediction Again: Is Attention All You Need?](https://arxiv.org/abs/2105.05563) :triangular_flag_on_post:**BOSS Zhipin**                                                                        | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/SAM)          | `torch`       |
| 35  | KDD'21            | [AOANet](./model_zoo/AOANet)             | [Architecture and Operation Adaptive Network for Online Recommendations](https://dl.acm.org/doi/10.1145/3447548.3467133) :triangular_flag_on_post:**Didi Chuxing**                                              | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/AOANet)       | `torch`       |
| 36  | AAAI'23           | [FinalMLP](./model_zoo/FinalMLP)         | [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction](https://arxiv.org/abs/2304.00902) :triangular_flag_on_post:**Huawei**                                                                                                               |     [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FinalMLP)         | `torch`       |
| 37  | SIGIR'23          | [FinalNet](./model_zoo/FinalNet)               | [FINAL: Factorized Interaction Layer for CTR Prediction](https://dl.acm.org/doi/10.1145/3539618.3591988) :triangular_flag_on_post:**Huawei**                                                                                                               |     [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/FinalNet)         | `torch`       |
| 38  | SIGIR'23          | [EulerNet](./model_zoo/EulerNet)               | [EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction](https://dl.acm.org/doi/10.1145/3539618.3591681) :triangular_flag_on_post:**Huawei**                                                                                                               |     [:arrow_upper_right:](https://github.com/Ethan-TZ/EulerNet/tree/main/%23Code4FuxiCTR%23)         | `torch`       |
| 39  | CIKM'23           | [GDCN](./model_zoo/GDCN)         | [Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction](https://dl.acm.org/doi/pdf/10.1145/3583780.3615089) :triangular_flag_on_post:**Microsoft**                                                                                                               |           | `torch`       |
| 40  | ICML'24          | [WuKong](./model_zoo/WuKong)               | [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545) :triangular_flag_on_post:**Meta**                                                        |       | `torch`       |
| 41  | Arxiv'24          | [DCNv3](./model_zoo/DCNv3)               | [DCNv3: Towards Next Generation Deep Cross Network for Click-Through Rate Prediction](https://arxiv.org/abs/2407.13349)                                               | [:arrow_upper_right:](https://github.com/salmon1802/DCNv3/tree/master/checkpoints)       | `torch`       |
|<tr><th colspan=6 align="center">:open_file_folder: **Behavior Sequence Modeling**</th></tr>|
| 42  | KDD'18            | [DIN](./model_zoo/DIN)                   | [Deep Interest Network for Click-Through Rate Prediction](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction) :triangular_flag_on_post:**Alibaba**        |   [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DIN)       | `torch`       |
| 43  | AAAI'19           | [DIEN](./model_zoo/DIEN)                 | [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672) :triangular_flag_on_post:**Alibaba**                                                                      |   [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DIEN)        | `torch`       |
| 44  | DLP-KDD'19        | [BST](./model_zoo/BST)                   | [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874) :triangular_flag_on_post:**Alibaba**                                                                 |  [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/BST)     | `torch`       |
| 45  | CIKM'20           | [DMIN](./model_zoo/DMIN)                 | [Deep Multi-Interest Network for Click-through Rate Prediction](https://dl.acm.org/doi/10.1145/3340531.3412092) :triangular_flag_on_post:**Alibaba**                                                            | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DMIN)                                                                                                                 | `torch`       |
| 46  | AAAI'20           | [DMR](./model_zoo/DMR)                   | [Deep Match to Rank Model for Personalized Click-Through Rate Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/5346) :triangular_flag_on_post:**Alibaba**                                           |    [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/DMR)                                                                                                                  | `torch`       |
| 47  | DLP-KDD'22          | [ETA](./model_zoo/LongCTR/ETA)                   | [Efficient Long Sequential User Data Modeling for Click-Through Rate Prediction](https://arxiv.org/abs/2209.12212) :triangular_flag_on_post:**Alibaba**                                                               |                                                                                                                 | `torch`       |
| 48  | CIKM'22           | [SDIM](./model_zoo/LongCTR/SDIM)                 | [Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction](https://arxiv.org/abs/2205.10249) :triangular_flag_on_post:**Meituan**                                                       |                                                                                                                 | `torch`       |
| 49  | KDD'23           | [TransAct](./model_zoo/TransAct)                 | [TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest](https://arxiv.org/abs/2306.00248) :triangular_flag_on_post:**Pinterest**                                                       | [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/TransAct)         | `torch`       |
|<tr><th colspan=6 align="center">:open_file_folder: **Dynamic Weight Network**</th></tr>|
| 50  | NeurIPS'22          | [APG](./model_zoo/APG)               | [APG: Adaptive Parameter Generation Network for Click-Through Rate Prediction](https://arxiv.org/abs/2203.16218) :triangular_flag_on_post:**Alibaba**                                |    [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/APG)                                                                                                   | `torch`       |
| 51  | KDD'23        | [PPNet](./model_zoo/PEPNet)             | [PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information](https://arxiv.org/abs/2302.01115) :triangular_flag_on_post:**KuaiShou**                          |    [:arrow_upper_right:](https://github.com/reczoo/BARS/tree/main/ranking/ctr/PPNet)                                                                                                   | `torch`       |
|<tr><th colspan=6 align="center">:open_file_folder: **Multi-Task Modeling**</th></tr>|
| 52  |     Arxiv'17      | [ShareBottom](./model_zoo/multitask/ShareBottom)               | [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098)                                                                                            |                                                                                                                 | `torch`       |
| 53  | KDD'18          | [MMoE](./model_zoo/multitask/MMOE)               | [Modeling Task Relationships in Multi-task Learning with Multi-Gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) :triangular_flag_on_post:**Google**                                                                                            |                                                                                                                 | `torch`       |
| 54  | RecSys'20          | [PLE](./model_zoo/multitask/PLE)               | [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236) :triangular_flag_on_post:**Tencent**                                                                                            |                                                                                                                 | `torch`       |


## Benchmarking

We have benchmarked FuxiCTR models on a set of open datasets as follows:

+ :star: [Benchmark datasets for CTR prediction](https://github.com/reczoo/Datasets?tab=readme-ov-file#ctr-prediction)
+ :star: [Benchmark settings and running steps](https://github.com/reczoo/BARS/tree/main/ranking/ctr)
+ :star: [Benchmark leaderboard for CTR prediction](https://openbenchmark.github.io/BARS/CTR/leaderboard)

## Dependencies

FuxiCTR has the following dependencies:

+ python 3.9+
+ pytorch 1.10+ (required only for torch models)
+ tensorflow 2.1+ (required only for tensorflow models)

Please install other required packages via `pip install -r requirements.txt`.

## Quick Start

1. Run the demo examples
   
    Examples are provided in the demo directory to show some basic usage of FuxiCTR. Users can run the examples for quick start and to understand the workflow. 
   
   ```
   cd demo
   python example1_build_dataset_to_parquet.py
   python example2_DeepFM_with_parquet_input.py
   ```

2. Run a model on tiny data
   
    Users can easily run each model in the model zoo following the commands below, which is a demo for running DCN. In addition, users can modify the dataset config and model config files to run on their own datasets or with new hyper-parameters. More details can be found in the [README](./model_zoo/DCN/DCN_torch/README.md).
   
   ```
   cd model_zoo/DCN/DCN_torch
   python run_expid.py --expid DCN_test --gpu 0

   # Change `MODEL` according to the target model name
   cd model_zoo/MODEL
   python run_expid.py --expid MODEL_test --gpu 0
   ```

3. Run a model on benchmark datasets (e.g., Criteo)

   Users can follow the [benchmark section](#Benchmarking) to get benchmark datasets and running steps for reproducing the existing results. Please see an example here: https://github.com/reczoo/BARS/tree/main/ranking/ctr/DCNv2/DCNv2_criteo_x1


4. Implement a new model
   
   The FuxiCTR library is designed to be modularized, so that every component can be overwritten by users according to their needs. In many cases, only the model class needs to be implemented for a new customized model. If data preprocessing or data loader is not directly applicable, one can also overwrite a new one through the [core APIs](https://www.processon.com/view/link/63cfcfab4e30670eac4a81c7). We show a concrete example which implements our new model [FinalMLP](https://reczoo.github.io/FinalMLP) that has been recently published in AAAI 2023.

5. Tune hyper-parameters of a model
   
   FuxiCTR currently support fast grid search of hyper-parameters of a model using multiple GPUs. The following example shows the grid search of 8 experiments with 4 GPUs.
    
   ```
   cd experiment
   python run_param_tuner.py --config config/DCN_tiny_parquet_tuner_config.yaml --gpu 0 1 2 3 0 1 2 3
   ```

## 🔥 Citation

If you find our code or benchmarks helpful in your research, please cite the following papers.

+ Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). *The 30th ACM International Conference on Information and Knowledge Management (CIKM)*, 2021. [[Bibtex](https://dblp.org/rec/conf/cikm/ZhuLYZH21.html?view=bibtex)]
+ Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, Rui Zhang. [BARS: Towards Open Benchmarking for Recommender Systems](https://arxiv.org/abs/2205.09626). *The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)*, 2022. [[Bibtex](https://dblp.org/rec/conf/sigir/ZhuDSMLCXZ22.html?view=bibtex)]

## Discussion

Welcome to join our WeChat group for any question and discussion. If you are interested in research and practice in recommender systems, please reach out via our WeChat group.

![Scan QR code](https://openbenchmark.github.io/BARS/_images/wechat.jpg)
