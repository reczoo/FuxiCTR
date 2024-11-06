#! /bin/sh
home="$(pwd)/../model_zoo"

echo "=== Testing AFM ==="  && cd $home/AFM && python run_expid.py --expid AFM_test && \
echo "=== Testing AFN ===" && cd $home/AFN && python run_expid.py --expid AFN_test && \
echo "=== Testing AOANet ===" && cd $home/AOANet && python run_expid.py --expid AOANet_test && \
echo "=== Testing APG ===" && cd $home/APG && python run_expid.py --expid APG_DeepFM_test && \
echo "=== Testing AutoInt ===" && cd $home/AutoInt && python run_expid.py --expid AutoInt_test && \
echo "=== Testing BST ===" && cd $home/BST && python run_expid.py --expid BST_test && \
echo "=== Testing CCPM ===" && cd $home/CCPM && python run_expid.py --expid CCPM_test && \
echo "=== Testing DCN ===" && cd $home/DCN/DCN_torch && python run_expid.py --expid DCN_test && \
echo "=== Testing DCNv2 ===" && cd $home/DCNv2 && python run_expid.py --expid DCNv2_test && \
echo "=== Testing DCNv3 ===" && cd $home/DCNv3 && python run_expid.py --expid DCNv3_test && \
echo "=== Testing DeepCrossing ===" && cd $home/DeepCrossing && python run_expid.py --expid DeepCrossing_test && \
echo "=== Testing DeepFM ===" && cd $home/DeepFM/DeepFM_torch && python run_expid.py --expid DeepFM_test && \
echo "=== Testing DeepIM ===" && cd $home/DeepIM && python run_expid.py --expid DeepIM_test && \
echo "=== Testing DESTINE ===" && cd $home/DESTINE && python run_expid.py --expid DESTINE_test && \
echo "=== Testing DIEN ===" && cd $home/DIEN && python run_expid.py --expid DIEN_test && \
echo "=== Testing DIN ===" && cd $home/DIN && python run_expid.py --expid DIN_test && \
echo "=== Testing DLRM ===" && cd $home/DLRM && python run_expid.py --expid DLRM_test && \
echo "=== Testing DMIN ===" && cd $home/DMIN && python run_expid.py --expid DMIN_test && \
echo "=== Testing DMR ===" && cd $home/DMR && python run_expid.py --expid DMR_test && \
echo "=== Testing DNN ===" && cd $home/DNN/DNN_torch && python run_expid.py --expid DNN_test && \
echo "=== Testing DSSM ===" && cd $home/DSSM && python run_expid.py --expid DSSM_test && \
echo "=== Testing EDCN ===" && cd $home/EDCN && python run_expid.py --expid EDCN_test && \
echo "=== Testing EulerNet ===" && cd $home/EulerNet && python run_expid.py --expid EulerNet_test && \
echo "=== Testing FFM ===" && cd $home/FFM && python run_expid.py --expid FFM_test && \
echo "=== Testing FFMv2 ===" && python run_expid.py --expid FFMv2_test 
echo "=== Testing FGCNN ===" && cd $home/FGCNN && python run_expid.py --expid FGCNN_test && \
echo "=== Testing FiBiNET ===" && cd $home/FiBiNET && python run_expid.py --expid FiBiNET_test && \
echo "=== Testing FiGNN ===" && cd $home/FiGNN && python run_expid.py --expid FiGNN_test && \
echo "=== Testing FinalMLP ===" && cd $home/FinalMLP && python run_expid.py --expid FinalMLP_test && \
echo "=== Testing FinalNet ===" && cd $home/FinalNet && python run_expid.py --expid FinalNet_test && \
echo "=== Testing FLEN ===" && cd $home/FLEN && python run_expid.py --expid FLEN_test && \
echo "=== Testing FM ===" && cd $home/FM && python run_expid.py --expid FM_test && \
echo "=== Testing FmFM ===" && cd $home/FmFM && python run_expid.py --expid FmFM_test && \
echo "=== Testing FwFM ===" && cd $home/FwFM && python run_expid.py --expid FwFM_test && \
echo "=== Testing GDCN ===" && cd $home/GDCN && python run_expid.py --expid GDCN_test && \
echo "=== Testing HFM ===" && cd $home/HFM && python run_expid.py --expid HFM_test && \
echo "=== Testing HOFM ===" && cd $home/HOFM && python run_expid.py --expid HOFM_test && \
echo "=== Testing InterHAt ===" && cd $home/InterHAt && python run_expid.py --expid InterHAt_test && \
echo "=== Testing LorentzFM ===" && cd $home/LorentzFM && python run_expid.py --expid LorentzFM_test && \
echo "=== Testing LR ===" && cd $home/LR && python run_expid.py --expid LR_test && \
echo "=== Testing MaskNet ===" && cd $home/MaskNet && python run_expid.py --expid MaskNet_test && \
echo "=== Testing NFM ===" && cd $home/NFM && python run_expid.py --expid NFM_test && \
echo "=== Testing ONN ===" && cd $home/ONN/ONN_torch && python run_expid.py --expid ONN_test && \
echo "=== Testing ONNv2 ===" && cd $home/ONN/ONN_torch && python run_expid.py --expid ONNv2_test && \
echo "=== Testing PPNet ===" && cd $home/PEPNet && python run_expid.py --expid PPNet_test && \
echo "=== Testing PNN ===" && cd $home/PNN && python run_expid.py --expid PNN_test && \
echo "=== Testing SAM ===" && cd $home/SAM && python run_expid.py --expid SAM_test && \
echo "=== Testing TransAct ===" && cd $home/TransAct && python run_expid.py --expid TransAct_test && \
echo "=== Testing WideDeep ===" && cd $home/WideDeep/WideDeep_torch && python run_expid.py --expid WideDeep_test && \
echo "=== Testing WuKong ===" && cd $home/WuKong && python run_expid.py --expid WuKong_test && \
echo "=== Testing xDeepFM ===" && cd $home/xDeepFM && python run_expid.py --expid xDeepFM_test && \

# Multi-task recommendation
echo "=== Testing ShareBottom ===" && cd $home/multitask/ShareBottom && python run_expid.py --expid ShareBottom_test && \
echo "=== Testing MMoE ===" && cd $home/multitask/MMoE && python run_expid.py --expid MMoE_test && \
echo "=== Testing PLE ===" && cd $home/multitask/PLE && python run_expid.py --expid PLE_test && \

echo "All tests done."
