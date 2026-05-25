## FuxiCTR Versions

### FuxiCTR v2.3
[Doing] Add support for saving pb file, exporting embeddings
[Doing] Add support of multi-gpu training

**FuxiCTR v2.3.9, 2025-06-17**
+ [FIX] Fixing preprocessing issues in v2.3.8.

**FuxiCTR v2.3.8, 2025-06-14** (Deprecated)
+ [FIX] Speed up data loading for large parquet files ([#140](https://github.com/reczoo/FuxiCTR/issues/140))
+ [FEA] Add ETA/SDIM/SIM/TWIN/MIRRN to LongCTR models
+ [RFR] Update WuKong for reproducing results
+ [FEA] Add example8 for demonstrating how to use embeding feature as input

**FuxiCTR v2.3.7, 2024-12-29**
+ [FIX] Fix regularization_loss() when feature_encoders exist ([#135](https://github.com/reczoo/FuxiCTR/issues/135))

**FuxiCTR v2.3.6, 2024-12-28**
+ [FIX] Fix init_weights() for PretrainedEmbedding by modifying embedding_initializer ([#126](https://github.com/reczoo/FuxiCTR/issues/126))
+ [FIX] Fix get_mask issue when num_heads > 1 ([#130](https://github.com/reczoo/FuxiCTR/issues/130))
+ [FIX]  Fix TransAct error when number of sequence features > 2 ([#132](https://github.com/reczoo/FuxiCTR/issues/132))

**FuxiCTR v2.3.5, 2024-11-06**
+ [FIX] Fix get_inputs() bug ([#115](https://github.com/reczoo/FuxiCTR/issues/115))

**FuxiCTR v2.3.4, 2024-11-05**
+ [FEA] Add WuKong model
+ [FIX] Fix OOV token update ([#119](https://github.com/reczoo/FuxiCTR/issues/119))
+ [FIX] Speed up parquet dataset reading ([#121](https://github.com/reczoo/FuxiCTR/issues/121))
+ [FIX] Fix add_loss() isue that does not work after renaming to compute_loss() ([#122](https://github.com/reczoo/FuxiCTR/issues/122))
+ [FIX] Rename customized reset_parameters to init_weights ([#123](https://github.com/reczoo/FuxiCTR/issues/123))

**FuxiCTR v2.3.3, 2024-10-14**
+ [FEA] Add EulerNet and DCNv3 models
+ [FEA] Add support to parquet as input, like csv format
+ [FIX] Add col_name as default args in feature_preprocess ([#105](https://github.com/reczoo/FuxiCTR/issues/105))
+ [FIX] Fix fill_na when copy from a column ([#117](https://github.com/reczoo/FuxiCTR/issues/117))

**FuxiCTR v2.3.2, 2024-07-11**
+ [FEA] Add TransAct model
+ [FEA] Add new feature type `embedding`, supporting [`meta`, `numeric`, `embedding`, `categorical`, `sequence`]
+ [FIX] Fix typo error in copy_from of version v2.3.1
+ [FIX] Fix issue in build_dataset for skipping rebuilding dataset
+ [FIX] Fix typo error in DIEN (AUGRUCell->AGRUCell)
+ [FIX] Fix typo error in feature_embedding of tf version ([#94](https://github.com/reczoo/FuxiCTR/issues/94))

**FuxiCTR v2.3.1, 2024-06-09**
+ [FIX] Fix customized preprossors based on polars and update demos
+ [DOC] Add copyrights
+ [FEA] Add `embedding_dim` setting to numeric features
  
**FuxiCTR v2.3.0, 2024-04-20**
+ [RFR] Support reading CSV and Parquet files as inputs
+ [FEA] Add dataloader for parquet
+ [FEA] Add the `rebuild_dataset=False` setting to skip rebuiding when the input dataset has already been preprocessed with ID feature mapping. This enables customized feature mapping instead of using FuxiCTR Preprocessor (which is slow for large dataset).

-------------------------------

### FuxiCTR v2.2

**FuxiCTR v2.2.3, 2024-04-20**
+ [FIX] Quick fix to v2.2.2 that miss one line when committing

**FuxiCTR v2.2.2, 2024-04-18 (Deprecated)**
+ [FEA] Update to use polars instead of pandas for faster feature processing
+ [FIX] When num_workers > 1, NpzBlockDataLoader cannot keep the reading order of samples ([#86](https://github.com/xue-pai/FuxiCTR/issues/86))

**FuxiCTR v2.2.1, 2024-04-16**
+ [FIX] Fix issue of evaluation not performed at epoch end when streaming=True ([#85](https://github.com/xue-pai/FuxiCTR/issues/85))
+ [FIX] Fix issue when loading pretrain_emb in npz format ([#84](https://github.com/xue-pai/FuxiCTR/issues/84))

**FuxiCTR v2.2.0, 2024-02-17**
+ [FEA] Add support of npz format for pretrained_emb
+ [RFR] Change data format from h5 to npz

-------------------------------

### FuxiCTR v2.1

**FuxiCTR v2.1.3, 2024-02-17**
+ [FEA] Add GDCN model
+ [RFR] Rename FINAL model to FinalNet
+ [RFR] Update RecZoo URLs
+ [FIX] Fix bug [#75](https://github.com/xue-pai/FuxiCTR/issues/75)
+ [FIX] Fix h5 file extenstion issue
+ [FIX] Fix typo in FinalNet
 
**FuxiCTR v2.1.2, 2023-11-01**
+ [RFR] Update H5DataBlockLoader to support dataloader with multiprocessing

**FuxiCTR v2.1.1, 2023-10-26**
+ [FEA] Update to allow loading pretrained h5 directly in PretrainedEmbedding (skip key mapping in preprocess)
+ [FEA] Update to allow data_path to be a directory path for h5

**FuxiCTR v2.1.0, 2023-10-23**
+ [FEA] Add PretrainedEmbedding Layer
+ [FEA] Update preprocess and features to support oov_idx based masking for PretrainedEmbedding
+ [FIX] Fix bug [#72](https://github.com/xue-pai/FuxiCTR/issues/72) for SDIM

-------------------------------

### FuxiCTR v2.0

**FuxiCTR v2.0.4, 2023-10-10**
+ [FEA] Add multi-task models (MMoE/PLE)
+ [FIX] Fix exception in run_expid.py when test_data is None

**FuxiCTR v2.0.3, 2023-05-14**
+ [FEA] Update DMIN, DMR, APG, PPNet, ONN_tf
+ [FIX] Change dynamic_emb_dim to flatten_emb

**FuxiCTR v2.0.2, 2023-05-14**
+ [FEA] Update FINAL, DIEN
+ [RFR] Update ordered_features to use_features

**FuxiCTR v2.0.1, 2023-02-15**
+ [DOC] Add fuxictr tutorials
+ [FEA] Update demo examples
+ [FIX] Fix build_dataset() to skip rebuilding if it already exists

**FuxiCTR v2.0.0, 2023-01-19**
+ [FEA] Add more models of year 2021-2022.
+ [FEA] Add tensorflow backbone support
+ [RFR] Refine code structure to support model development with minimal code

-------------------------------

### FuxiCTR v1.2

**FuxiCTR v1.2.2, 2022-07-03**
+ [FIX] Fix bug in EDCN ([#29](https://github.com/reczoo/FuxiCTR/issues/29))
+ [FIX] Fix MultiHeadAttention bug ([#30](https://github.com/reczoo/FuxiCTR/issues/30))

**FuxiCTR v1.2.1, 2022-06-12**
+ [FIX] Fix layernorm bug in MaskNet
+ [DOC] Refine demos and docs

**FuxiCTR v1.2.0, 2022-04-17**
+ [FEA] Add DSSM/DLRM/EDCN/AOANet/SAM models

-------------------------------

### FuxiCTR v1.1

**FuxiCTR v1.1.1, 2022-03-01**
+ [FEA] Add DESTINE/MaskNet models
+ [FEA] Add support for default FeatureEncoder on new datasets

**FuxiCTR v1.1.0, 2021-12-12**
+ [FEA] Refactor the code of layers.EmbeddingLayer
+ [FEA] Add new feature for loading blocks of h5 data
+ [FEA] Add tests for DIN, FmFM
+ [FEA] Add support for multiple fields concat for DIN
+ [RFR] Remove the unnecessary config of embedding_dropout because it does not help after some attempts
+ [FEA] Add embedding_hooks of dense layers on pretrained embeddings
+ [FIX] Fix the bug in padding_idx (have no effect on Criteo/Avazu results)
+ [FIX] Fix the bug in loading pretrained embeddings (have no effect on Criteo/Avazu results)
+ [DOC] Add tutorials on how to use sequence features and pretrained embeddings
  
-------------------------------

### FuxiCTR v1.0

**FuxiCTR v1.0.2, 2021-12-01**
+ [RFR] Refactor the code and documentation to support reproducing the BARS-CTR benchmark.

**FuxiCTR v1.0.1, 2021-10-01**
+ [FEA] The first release of FuxiCTR, including 28 models. This version was used for the CIKM'21 paper.
