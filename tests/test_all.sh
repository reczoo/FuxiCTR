#! /bin/sh

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


cd ../benchmarks

python run.py --expid AFM_test && \
python run.py --expid AFN_test && \
python run.py --expid AutoInt_test && \
python run.py --expid CCPM_test && \
python run.py --expid DCN_test && \
python run.py --expid DeepCrossing_test && \
python run.py --expid DeepFM_test && \
python run.py --expid DNN_test && \
python run.py --expid FFM_test && \
python run.py --expid FGCNN_test && \
python run.py --expid FiBiNET_test && \
python run.py --expid FiGNN_test && \
python run.py --expid FM_test && \
python run.py --expid FNN_test && \
python run.py --expid FwFM_test && \
python run.py --expid HFM_test && \
python run.py --expid HOFM_test && \
python run.py --expid InterHAt_test && \
python run.py --expid LorentzFM_test && \
python run.py --expid LR_test && \
python run.py --expid NFM_test && \
python run.py --expid ONN_test && \
python run.py --expid PNN_test && \
python run.py --expid WideDeep_test && \
python run.py --expid xDeepFM_test

echo "All tests done."
