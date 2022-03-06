## FuxiCTR Milestones


## FuxiCTR v2.0, Doing.
+ Add more models of year 2021.
+ Add support for saving pb file, exporting embeddings, exporting feature vocab


## FuxiCTR v1.1

### FuxiCTR v1.1.1, 2022-03-01
+ Add DESTINE/MaskNet models

### FuxiCTR v1.1.0, 2021-12-12.
+ Refactor the code of layers.EmbeddingLayer
+ Add support for loading blocks of h5 data
+ Add tests for DIN, FmFM
+ Refine the DIN model to support feature concatenation
+ Add tutorials on how to use sequence features and pretrained embeddings
+ Fix the defect in using padding_idx (no impact on Criteo/Avazu results)
+ Fix the defect in loading pretrain embeddings (no impact on Criteo/Avazu results)

## FuxiCTR v1.0

### FuxiCTR v1.0.2, 2021-12-01
Refactor the code and documentation to support reproducing steps on [the BARS benchmark](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks).

### FuxiCTR v1.0.1, 2021-10-01
This is the first release of FuxiCTR, including 28 models: LR, FM, CCPM, FFM, DNN, Wide&Deep, FNN, IPNN, DeepCross, HOFM, DeepFM, NFM, AFM, DCN, FwFM, xDeepFM, DIN, FiGNN, AutoInt+, FiBiNET, FGCNN, HFM+, ONN, AFN+, LorentzFM, InterHAt, FLEN, FmFM. Especially, this version corresponds to the original implementations used for reproducing the experiments in the following paper: *Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He, Open Benchmarking for CTR Prediction, CIKM 2021*.






