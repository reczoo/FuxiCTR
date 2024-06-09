#! /bin/sh
cd "$(pwd)/../demo"

echo "=== Testing example1 ===" && python example1_build_dataset_to_parquet.py && \
echo "=== Testing example2 ===" && python example2_DeepFM_with_parquet_input.py && \
echo "=== Testing example3 ===" && python example3_DeepFM_with_npz_input.py && \
echo "=== Testing example4 ===" && python example4_DeepFM_with_csv_input.py && \
echo "=== Testing example5 ===" && python example5_DeepFM_with_pretrained_emb.py && \
echo "=== Testing example6 ===" && python example6_DIN_with_sequence_feature.py && \
echo "=== Testing example7 ===" && python example7_DeepFM_with_customized_preprocess.py && \

echo "All tests done."
