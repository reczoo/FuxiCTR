#! /bin/sh

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

cd ../benchmarks

python benchmark.py --expid AFM_test --gpu 0 && \
python benchmark.py --expid AFN_test --gpu 0 && \
python benchmark.py --expid AutoInt_test --gpu 0 && \
python benchmark.py --expid CCPM_test --gpu 0 && \
python benchmark.py --expid DCN_test --gpu 0 && \
python benchmark.py --expid DeepCrossing_test --gpu 0 && \
python benchmark.py --expid DeepFM_test --gpu 0 && \
python benchmark.py --expid DNN_test --gpu 0 && \
python benchmark.py --expid FFM_test --gpu 0 && \
python benchmark.py --expid FGCNN_test --gpu 0 && \
python benchmark.py --expid FiBiNET_test --gpu 0 && \
python benchmark.py --expid FiGNN_test --gpu 0 && \
python benchmark.py --expid FM_test --gpu 0 && \
python benchmark.py --expid FNN_test --gpu 0 && \
python benchmark.py --expid FwFM_test --gpu 0 && \
python benchmark.py --expid HFM_test --gpu 0 && \
python benchmark.py --expid HOFM_test --gpu 0 && \
python benchmark.py --expid InterHAt_test --gpu 0 && \
python benchmark.py --expid LorentzFM_test --gpu 0 && \
python benchmark.py --expid LR_test --gpu 0 && \
python benchmark.py --expid NFM_test --gpu 0 && \
python benchmark.py --expid ONN_test --gpu 0 && \
python benchmark.py --expid PNN_test --gpu 0 && \
python benchmark.py --expid WideDeep_test --gpu 0 && \
python benchmark.py --expid xDeepFM_test --gpu 0

echo "All tests done."
