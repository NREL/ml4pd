import os
from sys import platform

from ml4pd._components import components
from ml4pd._registry import registry

# Try adding GraphViz executable to path for windows
if platform == "win32":
    current_paths = os.environ["PATH"].split(os.pathsep)
    for path in [r"C:\Program Files\Graphviz\bin", r"C:\Program Files (x86)\Graphviz\bin"]:
        if path not in current_paths:
            try:
                files = os.listdir(path)
                if "dot.exe" in files:
                    os.environ["PATH"] += os.pathsep + path
                    break
            except FileNotFoundError:
                pass
