#! /bin/sh
home="$(pwd)/../model_zoo"

echo "=== Testing AFM ==="  && cd $home/AFM && python run_expid.py --expid AFM_test && \
echo "=== Testing AFN ===" && cd $home/AFN && python run_expid.py --expid AFN_test && \
echo "=== Testing AOANet ===" && cd $home/AOANet && python run_expid.py --expid AOANet_test && \
echo "=== Testing AutoInt ===" && cd $home/AutoInt && python run_expid.py --expid AutoInt_test && \
echo "=== Testing BST ===" && cd $home/BST && python run_expid.py --expid BST_test && \
echo "=== Testing CCPM ===" && cd $home/CCPM && python run_expid.py --expid CCPM_test && \
echo "=== Testing DCN_torch ===" && cd $home/DCN/DCN_torch && python run_expid.py --expid DCN_test && \
echo "=== Testing DCNv2 ===" && cd $home/DCNv2 && python run_expid.py --expid DCNv2_test && \
echo "=== Testing DeepCrossing ===" && cd $home/DeepCrossing && python run_expid.py --expid DeepCrossing_test && \
echo "=== Testing DeepFM_torch ===" && cd $home/DeepFM/DeepFM_torch && python run_expid.py --expid DeepFM_test && \
echo "=== Testing DeepFM_tf ===" && cd $home/DeepFM/DeepFM_tf && python run_expid.py --expid DeepFM_test && \
echo "=== Testing DeepIM ===" && cd $home/DeepIM && python run_expid.py --expid DeepIM_test && \
echo "=== Testing DESTINE ===" && cd $home/DESTINE && python run_expid.py --expid DESTINE_test && \
echo "=== Testing DIEN ===" && cd $home/DIEN && python run_expid.py --expid DIEN_test && \
echo "=== Testing DIN ===" && cd $home/DIN && python run_expid.py --expid DIN_test && \
echo "=== Testing DMIN ===" && cd $home/DMIN && python run_expid.py --expid DMIN_test && \
echo "=== Testing DMR ===" && cd $home/DMR && python run_expid.py --expid DMR_test && \
echo "=== Testing DLRM ===" && cd $home/DLRM && python run_expid.py --expid DLRM_test && \
echo "=== Testing DNN_torch ===" && cd $home/DNN/DNN_torch && python run_expid.py --expid DNN_test && \
echo "=== Testing DSSM ===" && cd $home/DSSM && python run_expid.py --expid DSSM_test && \
echo "=== Testing EDCN ===" && cd $home/EDCN && python run_expid.py --expid EDCN_test && \
echo "=== Testing FFM ===" && cd $home/FFM && python run_expid.py --expid FFM_test && \
echo "=== Testing FFMv2 ===" && python run_expid.py --expid FFMv2_test 
echo "=== Testing FGCNN ===" && cd $home/FGCNN && python run_expid.py --expid FGCNN_test && \
echo "=== Testing DNN ===" && cd $home/FiBiNET && python run_expid.py --expid FiBiNET_test && \
echo "=== Testing FiGNN ===" && cd $home/FiGNN && python run_expid.py --expid FiGNN_test && \
echo "=== Testing FLEN ===" && cd $home/FLEN && python run_expid.py --expid FLEN_test && \
echo "=== Testing FM ===" && cd $home/FM && python run_expid.py --expid FM_test && \
echo "=== Testing FmFM ===" && cd $home/FmFM && python run_expid.py --expid FmFM_test && \
echo "=== Testing FwFM ===" && cd $home/FwFM && python run_expid.py --expid FwFM_test && \
echo "=== Testing FinalMLP ===" && cd $home/FinalMLP && python run_expid.py --expid FinalMLP_test && \
# echo "=== Testing FINAL ===" && cd $home/FINAL && python run_expid.py --expid FINAL_test && \
echo "=== Testing HFM ===" && cd $home/HFM && python run_expid.py --expid HFM_test && \
echo "=== Testing HOFM ===" && cd $home/HOFM && python run_expid.py --expid HOFM_test && \
echo "=== Testing InterHAt ===" && cd $home/InterHAt && python run_expid.py --expid InterHAt_test && \
echo "=== Testing LorentzFM ===" && cd $home/LorentzFM && python run_expid.py --expid LorentzFM_test && \
echo "=== Testing LR ===" && cd $home/LR && python run_expid.py --expid LR_test && \
echo "=== Testing MaskNet ===" && cd $home/MaskNet && python run_expid.py --expid MaskNet_test && \
echo "=== Testing NFM ===" && cd $home/NFM && python run_expid.py --expid NFM_test && \
echo "=== Testing ONN ===" && cd $home/ONN && python run_expid.py --expid ONN_test && \
echo "=== Testing ONNv2 ===" && python run_expid.py --expid ONNv2_test && \
echo "=== Testing PNN ===" && cd $home/PNN && python run_expid.py --expid PNN_test && \
echo "=== Testing SAM ===" && cd $home/SAM && python run_expid.py --expid SAM_test && \
echo "=== Testing WideDeep_torch ===" && cd $home/WideDeep/WideDeep_torch && python run_expid.py --expid WideDeep_test && \
echo "=== Testing xDeepFM ===" && cd $home/xDeepFM && python run_expid.py --expid xDeepFM_test && \

echo "All tests done."
