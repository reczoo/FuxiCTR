## FuxiCTR Versions

### FuxiCTR v2.1
[Doing] Add support for saving pb file, exporting embeddings


**FuxiCTR v2.1.2, 2023-11-01**
+ [Feature] Update H5DataBlockLoader to support dataloader with multiprocessing

**FuxiCTR v2.1.1, 2023-10-26**
+ [Feature] Update to allow loading pretrained h5 directly in PretrainedEmbedding (skip key mapping in preprocess)
+ [Feature] Update to allow data_path to be a directory path for h5

**FuxiCTR v2.1.0, 2023-10-23**
+ [Feature] Add PretrainedEmbedding Layer
+ [Feature] Update preprocess and features to support oov_idx based masking for PretrainedEmbedding
+ [Fix] Fix bug [#72](https://github.com/xue-pai/FuxiCTR/issues/72) for SDIM

-------------------------------

### FuxiCTR v2.0

**FuxiCTR v2.0.4, 2023-10-10**
+ [Feature] Add multi-task models (MMoE/PLE)
+ [Fix] Fix exception in run_expid.py when test_data is None

**FuxiCTR v2.0.3, 2023-05-14**
+ [Feature] Update DMIN, DMR, APG, PPNet, ONN_tf
+ [Fix] Change dynamic_emb_dim to flatten_emb

**FuxiCTR v2.0.2, 2023-05-14**
+ [Feature] Update FINAL, DIEN
+ [Feature] Update ordered_features to use_features

**FuxiCTR v2.0.1, 2023-02-15**
+ [Doc] Add fuxictr tutorials
+ [Feature] Update demo examples
+ [Fix] Fix build_dataset() to skip rebuilding if it already exists

**FuxiCTR v2.0.0, 2023-01-19**
+ [Feature] Add more models of year 2021-2022.
+ [Feature] dd tensorflow backbone support
+ [Feature] Refine code structure to support model development with minimal code

-------------------------------

### FuxiCTR v1.2

**FuxiCTR v1.2.2, 2022-07-03**
+ [Fix] Fix bug in EDCN #29
+ [Fix] Fix MultiHeadAttention bug #30

**FuxiCTR v1.2.1, 2022-06-12**
+ [Fix] Fix layernorm bug in MaskNet
+ [Doc] Refine demos and docs

**FuxiCTR v1.2.0, 2022-04-17**
+ [Feature] Add DSSM/DLRM/EDCN/AOANet/SAM models

-------------------------------

### FuxiCTR v1.1

**FuxiCTR v1.1.1, 2022-03-01**
+ [Feature] Add DESTINE/MaskNet models
+ [Feature] Add support for default FeatureEncoder on new datasets

**FuxiCTR v1.1.0, 2021-12-12**
+ [Feature] Refactor the code of layers.EmbeddingLayer
+ [Feature] Add new feature for loading blocks of h5 data
+ [Feature] Add tests for DIN, FmFM
+ [Feature] Add support for multiple fields concat for DIN
+ [Feature] Remove the unnecessary config of embedding_dropout because it does not help after some attempts
+ [Feature] Add embedding_hooks of dense layers on pretrained embeddings
+ [Fix] Fix the bug in padding_idx (have no effect on Criteo/Avazu results)
+ [Fix] Fix the bug in loading pretrained embeddings (have no effect on Criteo/Avazu results)
+ [Doc] Add tutorials on how to use sequence features and pretrained embeddings
  
-------------------------------

### FuxiCTR v1.0

**FuxiCTR v1.0.2, 2021-12-01**
+ [Feature] Refactor the code and documentation to support reproducing the BARS-CTR benchmark.

**FuxiCTR v1.0.1, 2021-10-01**
+ [Feature] The first release of FuxiCTR, including 28 models. This version was used for running the experiments in the CIKM'21 paper.
