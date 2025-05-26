# Run in first cell of notebooks: %run setup.py or `pip install -e` from project root

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))
    
    