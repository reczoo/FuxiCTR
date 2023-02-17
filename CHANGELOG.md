## FuxiCTR Milestones

### FuxiCTR v2.0

[Doing] Add support for saving pb file, exporting embeddings, exporting feature vocab

**FuxiCTR v2.0.1, 2023-02-15**

Docs
    + Add fuxictr tutorials

Fixes
    + Update demo examples
    + Fix build_dataset() to skip rebuilding if it already exists

**FuxiCTR v2.0.0, 2023-01-19**

Features
    - Add more models of year 2021-2022.
    - Add tensorflow backbone support
    - Refine code structure to support model development with minimal code

-------------------------------

### FuxiCTR v1.2

**FuxiCTR v1.2.2, 2022-07-03**

+ Fix bug in EDCN #29

+ Fix MultiHeadAttention bug #30

**FuxiCTR v1.2.1, 2022-06-12**

+ Fix layernorm bug in MaskNet
+ Refine demos and docs

**FuxiCTR v1.2.0, 2022-04-17**

+ Add DSSM/DLRM/EDCN/AOANet/SAM models

-------------------------------

### FuxiCTR v1.1

**FuxiCTR v1.1.1, 2022-03-01**

+ Add DESTINE/MaskNet models
+ Add support for default FeatureEncoder on new datasets

**FuxiCTR v1.1.0, 2021-12-12**

+ Refactor the code of layers.EmbeddingLayer
+ Add new feature for loading blocks of h5 data
+ Add tests for DIN, FmFM
+ Add support for multiple fields concat for DIN
+ Add tutorials on how to use sequence features and pretrained embeddings
+ Fix the defect in padding_idx (no impact on Criteo/Avazu results)
+ Fix the defect in loading pretrained embeddings (no impact on Criteo/Avazu results)
+ Remove the unnecessary config of embedding_dropout because it does not help after some attempts
+ Add embedding_hooks of dense layers on pretrained embeddings
  
-------------------------------

### FuxiCTR v1.0

**FuxiCTR v1.0.2, 2021-12-01**

+ Refactor the code and documentation to support reproducing steps on [the BARS benchmark](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks).

**FuxiCTR v1.0.1, 2021-10-01**

+ This is the first release of FuxiCTR, including 28 models: LR, FM, CCPM, FFM, DNN, Wide&Deep, FNN, IPNN, DeepCross, HOFM, DeepFM, NFM, AFM, DCN, FwFM, xDeepFM, DIN, FiGNN, AutoInt+, FiBiNET, FGCNN, HFM+, ONN, AFN+, LorentzFM, InterHAt, FLEN, FmFM. Especially, this version corresponds to the original implementations used for reproducing the experiments in the following paper: *Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He, Open Benchmarking for CTR Prediction, CIKM 2021*.
