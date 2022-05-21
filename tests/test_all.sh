#! /bin/sh

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
python run_expid.py --expid xDeepFM_test && \
python run_expid.py --expid DCNv2_test && \
python run_expid.py --expid FFMv2_test && \
python run_expid.py --expid ONNv2_test && \
python run_expid.py --expid FmFM_test && \
python run_expid.py --expid DeepIM_test && \
python run_expid.py --expid FLEN_test && \
python run_expid.py --expid DIN_test && \
python run_expid.py --expid DESTINE_test && \
python run_expid.py --expid MaskNet_test && \
python run_expid.py --expid DSSM_test && \
python run_expid.py --expid AOANet_test && \
python run_expid.py --expid DLRM_test && \
python run_expid.py --expid SAM_test && \
python run_expid.py --expid EDCN_test

echo "All tests done."
