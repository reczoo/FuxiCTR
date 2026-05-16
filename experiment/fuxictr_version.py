# pip install -U fuxictr
import sys
sys.path.append("../")
import fuxictr
from packaging import version
assert version.parse(fuxictr.__version__) >= version.parse("2.3.7")
