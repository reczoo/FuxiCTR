#! /bin/sh
home="$(pwd)/../model_zoo"

echo "=== Testing DNN_tf ===" && cd $home/DNN/DNN_tf && python run_expid.py --expid DNN_test && \
echo "=== Testing DCN_tf ===" && cd $home/DCN/DCN_tf && python run_expid.py --expid DCN_test && \
echo "=== Testing WideDeep_tf ===" && cd $home/WideDeep/WideDeep_tf && python run_expid.py --expid WideDeep_test && \

echo "All tests done."
