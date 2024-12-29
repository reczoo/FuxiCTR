## FuxiCTR Versions

### FuxiCTR v2.3
[Doing] Add support for saving pb file, exporting embeddings
[Doing] Add support of multi-gpu training

**FuxiCTR v2.3.7, 2024-12-29**
+ [Fix] Fix regularization_loss() when feature_encoders exist ([#135](https://github.com/reczoo/FuxiCTR/issues/135))

**FuxiCTR v2.3.6, 2024-12-28**
+ [Fix] Fix init_weights() for PretrainedEmbedding by modifying embedding_initializer ([#126](https://github.com/reczoo/FuxiCTR/issues/126))
+ [Fix] Fix get_mask issue when num_heads > 1 ([#130](https://github.com/reczoo/FuxiCTR/issues/130))
+ [Fix]  Fix TransAct error when number of sequence features > 2 ([#132](https://github.com/reczoo/FuxiCTR/issues/132))

**FuxiCTR v2.3.5, 2024-11-06**
+ [Fix] Fix get_inputs() bug ([#115](https://github.com/reczoo/FuxiCTR/issues/115))

**FuxiCTR v2.3.4, 2024-11-05**
+ [Feature] Add WuKong model
+ [Fix] Fix OOV token update ([#119](https://github.com/reczoo/FuxiCTR/issues/119))
+ [Fix] Speed up parquet dataset reading ([#121](https://github.com/reczoo/FuxiCTR/issues/121))
+ [Fix] Fix add_loss() isue that does not work after renaming to compute_loss() ([#122](https://github.com/reczoo/FuxiCTR/issues/122))
+ [Fix] Rename customized reset_parameters to init_weights ([#123](https://github.com/reczoo/FuxiCTR/issues/123))

**FuxiCTR v2.3.3, 2024-10-14**
+ [Feature] Add EulerNet and DCNv3 models
+ [Feature] Add support to parquet as input, like csv format
+ [Fix] Add col_name as default args in feature_preprocess ([#105](https://github.com/reczoo/FuxiCTR/issues/105))
+ [Fix] Fix fill_na when copy from a column ([#117](https://github.com/reczoo/FuxiCTR/issues/117))

**FuxiCTR v2.3.2, 2024-07-11**
+ [Feature] Add TransAct model
+ [Feature] Add new feature type `embedding`, supporting [`meta`, `numeric`, `embedding`, `categorical`, `sequence`]
+ [Fix] Fix typo error in copy_from of version v2.3.1
+ [Fix] Fix issue in build_dataset for skipping rebuilding dataset
+ [Fix] Fix typo error in DIEN (AUGRUCell->AGRUCell)
+ [Fix] Fix typo error in feature_embedding of tf version ([#94](https://github.com/reczoo/FuxiCTR/issues/94))

**FuxiCTR v2.3.1, 2024-06-09**
+ [Fix] Fix customized preprossors based on polars and update demos
+ [Doc] Add copyrights
+ [Feature] Add `embedding_dim` setting to numeric features
  
**FuxiCTR v2.3.0, 2024-04-20**
+ [Refactor] Support reading CSV and Parquet files as inputs
+ [Feature] Add dataloader for parquet
+ [Feature] Add the `rebuild_dataset=False` setting to skip rebuiding when the input dataset has already been preprocessed with ID feature mapping. This enables customized feature mapping instead of using FuxiCTR Preprocessor (which is slow for large dataset).

-------------------------------

### FuxiCTR v2.2

**FuxiCTR v2.2.3, 2024-04-20**
+ [Fix] Quick fix to v2.2.2 that miss one line when committing

**FuxiCTR v2.2.2, 2024-04-18 (Deprecated)**
+ [Feature] Update to use polars instead of pandas for faster feature processing
+ [Fix] When num_workers > 1, NpzBlockDataLoader cannot keep the reading order of samples ([#86](https://github.com/xue-pai/FuxiCTR/issues/86))

**FuxiCTR v2.2.1, 2024-04-16**
+ [Fix] Fix issue of evaluation not performed at epoch end when streaming=True ([#85](https://github.com/xue-pai/FuxiCTR/issues/85))
+ [Fix] Fix issue when loading pretrain_emb in npz format ([#84](https://github.com/xue-pai/FuxiCTR/issues/84))

**FuxiCTR v2.2.0, 2024-02-17**
+ [Feature] Add support of npz format for pretrained_emb
+ [Refactor] Change data format from h5 to npz

-------------------------------

### FuxiCTR v2.1

**FuxiCTR v2.1.3, 2024-02-17**
+ [Feature] Add GDCN model
+ [Refactor] Rename FINAL model to FinalNet
+ [Refactor] Update RecZoo URLs
+ [Fix] Fix bug [#75](https://github.com/xue-pai/FuxiCTR/issues/75)
+ [Fix] Fix h5 file extenstion issue
+ [Fix] Fix typo in FinalNet
 
**FuxiCTR v2.1.2, 2023-11-01**
+ [Refactor] Update H5DataBlockLoader to support dataloader with multiprocessing

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
+ [Refactor] Update ordered_features to use_features

**FuxiCTR v2.0.1, 2023-02-15**
+ [Doc] Add fuxictr tutorials
+ [Feature] Update demo examples
+ [Fix] Fix build_dataset() to skip rebuilding if it already exists

**FuxiCTR v2.0.0, 2023-01-19**
+ [Feature] Add more models of year 2021-2022.
+ [Feature] Add tensorflow backbone support
+ [Refactor] Refine code structure to support model development with minimal code

-------------------------------

### FuxiCTR v1.2

**FuxiCTR v1.2.2, 2022-07-03**
+ [Fix] Fix bug in EDCN ([#29](https://github.com/reczoo/FuxiCTR/issues/29))
+ [Fix] Fix MultiHeadAttention bug ([#30](https://github.com/reczoo/FuxiCTR/issues/30))

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
+ [Refactor] Remove the unnecessary config of embedding_dropout because it does not help after some attempts
+ [Feature] Add embedding_hooks of dense layers on pretrained embeddings
+ [Fix] Fix the bug in padding_idx (have no effect on Criteo/Avazu results)
+ [Fix] Fix the bug in loading pretrained embeddings (have no effect on Criteo/Avazu results)
+ [Doc] Add tutorials on how to use sequence features and pretrained embeddings
  
-------------------------------

### FuxiCTR v1.0

**FuxiCTR v1.0.2, 2021-12-01**
+ [Refactor] Refactor the code and documentation to support reproducing the BARS-CTR benchmark.

**FuxiCTR v1.0.1, 2021-10-01**
+ [Feature] The first release of FuxiCTR, including 28 models. This version was used for the CIKM'21 paper.
