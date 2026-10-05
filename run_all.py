# runs every demo in lesson order. first run downloads glove (~80 mb), bert (~440 mb) and minilm (~90 mb).

import subprocess
import sys
from pathlib import Path

for script in sorted(Path(__file__).parent.glob("0*.py")):
    print(f"\n>>> {script.name}")
    subprocess.run([sys.executable, str(script)], check=True)
