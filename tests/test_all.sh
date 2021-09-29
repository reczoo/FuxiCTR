#! /bin/sh

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


cd ../benchmarks

python benchmark.py --expid AFM_test && \
python benchmark.py --expid AFN_test && \
python benchmark.py --expid AutoInt_test && \
python benchmark.py --expid CCPM_test && \
python benchmark.py --expid DCN_test && \
python benchmark.py --expid DeepCrossing_test && \
python benchmark.py --expid DeepFM_test && \
python benchmark.py --expid DNN_test && \
python benchmark.py --expid FFM_test && \
python benchmark.py --expid FGCNN_test && \
python benchmark.py --expid FiBiNET_test && \
python benchmark.py --expid FiGNN_test && \
python benchmark.py --expid FM_test && \
python benchmark.py --expid FNN_test && \
python benchmark.py --expid FwFM_test && \
python benchmark.py --expid HFM_test && \
python benchmark.py --expid HOFM_test && \
python benchmark.py --expid InterHAt_test && \
python benchmark.py --expid LorentzFM_test && \
python benchmark.py --expid LR_test && \
python benchmark.py --expid NFM_test && \
python benchmark.py --expid ONN_test && \
python benchmark.py --expid PNN_test && \
python benchmark.py --expid WideDeep_test && \
python benchmark.py --expid xDeepFM_test

echo "All tests done."
