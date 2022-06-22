import json
import os
from pathlib import Path

current_dir = Path(__file__).absolute().parent.resolve()


def clear_text_output(notebook, pattern: str):
    cells = notebook["cells"]

    for cell_no, cell in enumerate(cells):
        if "outputs" in cell:
            for output_no, output in enumerate(cell["outputs"]):
                if "text" in output:
                    for sub_text in output["text"]:
                        if pattern in sub_text:
                            text = notebook["cells"][cell_no]["outputs"][output_no]["text"][output_no]
                            notebook["cells"][cell_no]["outputs"][output_no]["text"][output_no] = text.replace(pattern, "")
    return notebook


patterns = [
    r'Warning: Could not load "C:\\Users\\Hvo\\Miniconda3\\envs\\ml4pd\\Library\\bin\\gvplugin_pango.dll" - It was found, so perhaps one of its dependents was not.  Try ldd.',
    r'Warning: Could not load "C:\Users\hvo\Anaconda3\envs\ml4pd\Library\bin\gvplugin_pango.dll" - It was found, so perhaps one of its dependents was not.  Try ldd.',
]

notebooks = [current_dir / f"tutorial/{fname}" for fname in os.listdir(current_dir / "tutorial") if ".ipynb" in fname]

for notebook in notebooks:

    with open(notebook, mode="r", encoding="utf-8") as fname:
        json_file = json.load(fname)

    for pattern in patterns:
        new_json_file = clear_text_output(json_file, pattern)

    with open(notebook, mode="w") as fname:
        json.dump(new_json_file, fname)
