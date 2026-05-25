# pip install fuxictr
import fuxictr
from packaging import version
assert version.parse(fuxictr.__version__) >= version.parse("2.0.0")
