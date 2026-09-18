import os
import sys

# SAM2 is vendored unmodified under third_party/ and imports itself by the
# absolute name `sam2`, so third_party/ must be on the path.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "third_party"))
