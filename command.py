import subprocess
import sys
from pathlib import Path

"""
This is the main file to execute the commands.
It does both generates the new conversation between the debtor and agent and
calculates the similarity between the generated conversation and the ones conducted by humans.
Before running this file, make sure that conversation.py and similarities.py are in
the same directory as this script, and that a .env file with OPENAI_API_KEY exists.
"""

BASE_DIR = Path(__file__).resolve().parent
python_exe = sys.executable

# Run conversation.py
subprocess.run([python_exe, str(BASE_DIR / "conversation.py")])

# Run similarities.py
subprocess.run([python_exe, str(BASE_DIR / "similarities.py")])
