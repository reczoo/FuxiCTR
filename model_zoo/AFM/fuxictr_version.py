# pip install -U fuxictr
import fuxictr
from packaging import version
print(fuxictr.__version__)
assert version.parse(fuxictr.__version__) >= version.parse("2.3.7")
