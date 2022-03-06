#! /bin/sh

cd ../benchmarks

python run_expid.py --expid AFM_test --gpu 0 && \
python run_expid.py --expid AFN_test --gpu 0 && \
python run_expid.py --expid AutoInt_test --gpu 0 && \
python run_expid.py --expid CCPM_test --gpu 0 && \
python run_expid.py --expid DCN_test --gpu 0 && \
python run_expid.py --expid DeepCrossing_test --gpu 0 && \
python run_expid.py --expid DeepFM_test --gpu 0 && \
python run_expid.py --expid DNN_test --gpu 0 && \
python run_expid.py --expid FFM_test --gpu 0 && \
python run_expid.py --expid FGCNN_test --gpu 0 && \
python run_expid.py --expid FiBiNET_test --gpu 0 && \
python run_expid.py --expid FiGNN_test --gpu 0 && \
python run_expid.py --expid FM_test --gpu 0 && \
python run_expid.py --expid FNN_test --gpu 0 && \
python run_expid.py --expid FwFM_test --gpu 0 && \
python run_expid.py --expid HFM_test --gpu 0 && \
python run_expid.py --expid HOFM_test --gpu 0 && \
python run_expid.py --expid InterHAt_test --gpu 0 && \
python run_expid.py --expid LorentzFM_test --gpu 0 && \
python run_expid.py --expid LR_test --gpu 0 && \
python run_expid.py --expid NFM_test --gpu 0 && \
python run_expid.py --expid ONN_test --gpu 0 && \
python run_expid.py --expid PNN_test --gpu 0 && \
python run_expid.py --expid WideDeep_test --gpu 0 && \
python run_expid.py --expid xDeepFM_test --gpu 0
python run_expid.py --expid FmFM_test --gpu 0 && \
python run_expid.py --expid DCNv2_test --gpu 0 && \
python run_expid.py --expid ONNv2_test --gpu 0 && \
python run_expid.py --expid FFMv2_test --gpu 0 && \
python run_expid.py --expid DeepIM_test --gpu 0 && \
python run_expid.py --expid FLEN_test --gpu 0 && \
python run_expid.py --expid DIN_test --gpu 0 && \
python run_expid.py --expid DESTINE_test --gpu 0 && \
python run_expid.py --expid MaskNet_test --gpu 0 

echo "All tests done."
