# Quick Start

FuxiCTR supports two types of usages as follows.

1. **To run an existing model**: Users can easily run each model in the model zoo following the commands below, which is a demo for running DCN. In addition, users can modify the dataset config and model config files to run on their own datasets or with new hyper-parameters. More details can be found in the [readme file](https://github.com/xue-pai/FuxiCTR/blob/v2.0.0/model_zoo/DCN/DCN_torch/README.md).
    ```
    cd model_zoo/DCN/DCN_torch
    python run_expid.py --expid DCN_test --gpu 0
    ```

2. **To implement a new model**: The FuxiCTR code structure is modularized, so that every part can be overwritten by users according to their needs. As the workflow shown in the following figure, the orange parts comprise the minimal user code to implement a new customized model. In case that data preprocessing or data loader is not directly applicable, one can overwrite a new one through the [core APIs](https://www.processon.com/view/link/63cfcfab4e30670eac4a81c7). Some examples can also be found in the model zoo.

    <div align="center">
    <img src="https://cdn.jsdelivr.net/gh/xue-pai/FuxiCTR@main/docs/workflow.jpg" alt="FuxiCTR Workflow"/>
    </div>



