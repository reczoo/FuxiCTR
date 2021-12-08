#! /bin/sh

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


cd ../benchmarks

python run_expid.py --expid AFM_test && \
python run_expid.py --expid AFN_test && \
python run_expid.py --expid AutoInt_test && \
python run_expid.py --expid CCPM_test && \
python run_expid.py --expid DCN_test && \
python run_expid.py --expid DeepCrossing_test && \
python run_expid.py --expid DeepFM_test && \
python run_expid.py --expid DNN_test && \
python run_expid.py --expid FFM_test && \
python run_expid.py --expid FGCNN_test && \
python run_expid.py --expid FiBiNET_test && \
python run_expid.py --expid FiGNN_test && \
python run_expid.py --expid FM_test && \
python run_expid.py --expid FNN_test && \
python run_expid.py --expid FwFM_test && \
python run_expid.py --expid HFM_test && \
python run_expid.py --expid HOFM_test && \
python run_expid.py --expid InterHAt_test && \
python run_expid.py --expid LorentzFM_test && \
python run_expid.py --expid LR_test && \
python run_expid.py --expid NFM_test && \
python run_expid.py --expid ONN_test && \
python run_expid.py --expid PNN_test && \
python run_expid.py --expid WideDeep_test && \
python run_expid.py --expid xDeepFM_test

echo "All tests done."
