import sys
from pathlib import Path

# Make `import src...` work when pytest is run from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
