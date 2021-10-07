#! /bin/sh

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

cd ../benchmarks

python run.py --expid AFM_test --gpu 0 && \
python run.py --expid AFN_test --gpu 0 && \
python run.py --expid AutoInt_test --gpu 0 && \
python run.py --expid CCPM_test --gpu 0 && \
python run.py --expid DCN_test --gpu 0 && \
python run.py --expid DeepCrossing_test --gpu 0 && \
python run.py --expid DeepFM_test --gpu 0 && \
python run.py --expid DNN_test --gpu 0 && \
python run.py --expid FFM_test --gpu 0 && \
python run.py --expid FGCNN_test --gpu 0 && \
python run.py --expid FiBiNET_test --gpu 0 && \
python run.py --expid FiGNN_test --gpu 0 && \
python run.py --expid FM_test --gpu 0 && \
python run.py --expid FNN_test --gpu 0 && \
python run.py --expid FwFM_test --gpu 0 && \
python run.py --expid HFM_test --gpu 0 && \
python run.py --expid HOFM_test --gpu 0 && \
python run.py --expid InterHAt_test --gpu 0 && \
python run.py --expid LorentzFM_test --gpu 0 && \
python run.py --expid LR_test --gpu 0 && \
python run.py --expid NFM_test --gpu 0 && \
python run.py --expid ONN_test --gpu 0 && \
python run.py --expid PNN_test --gpu 0 && \
python run.py --expid WideDeep_test --gpu 0 && \
python run.py --expid xDeepFM_test --gpu 0

echo "All tests done."
