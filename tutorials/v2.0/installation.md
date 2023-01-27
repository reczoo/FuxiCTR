# Install FuxiCTR

```{note}
Tutorials for FuxiCTR v2 only.
```

FuxiCTR v2 has the following requirements.

+ pytorch 1.10+ (required only for torch models)
+ tensorflow 2.1+ (required only for tf models)
+ python 3.6+
+ pyyaml 5.1+
+ scikit-learn
+ pandas
+ numpy
+ h5py
+ tqdm

We recommend to install the above enviornment with `python 3.7` through Anaconda using [Anaconda3-2020.02-Linux-x86_64.sh](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh). 

For pytorch, you can download the appropriate whl file according to your CUDA version from https://download.pytorch.org/whl/torch_stable.html, and install offline. For eaxmple:

```
pip install torch-1.11.0%2Bcu102-cp37-cp37m-linux_x86_64.whl
```

There are two ways to install FuxiCTR v2:

**Solution 1**: pip install

```
pip install fuxictr==2.0.0
```

```{note}
All the dependent packages need to be installed accordingly.
```

**Solution 2**: git clone or download the zip file: https://github.com/xue-pai/FuxiCTR/tags

If you download the source code, you need to add the fuxictr folder to the system path in your code.

```python
import sys
sys.path.append('./YOUR_PATH_TO_FuxiCTR')
```

Check if fuxictr has been installed successfully.

```python
import fuxictr
print(fuxictr.__version__)
```
