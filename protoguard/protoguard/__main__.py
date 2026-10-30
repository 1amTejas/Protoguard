"""Allow ``python -m protoguard``."""
import sys

from .cli import main

sys.exit(main())
