import os
from pathlib import Path
from ml4pd.streams import MaterialStream
from ml4pd.aspen_units import Distillation, Flash
from ml4pd._components import Components
from ml4pd.flowsheet import Flowsheet

current_dir = Path(__file__).absolute().parent.resolve()

def docstr2md(docstring: str, md_file: str, title: str = None):

    lines = docstring.split("\n")
    lines = [line.replace("    ", "", 1) for line in lines]

    if title is not None:
        lines = [f"# {title}"] + lines

    with open(md_file, "w+") as fname:
        fname.writelines(line + "\n" for line in lines)


docstr2md(MaterialStream.__doc__, current_dir/'references/streams/material-stream.md', "Material Stream")
docstr2md(Distillation.__doc__, current_dir/'references/columns/distillation.md', "RadFrac")
docstr2md(Flash.__doc__, current_dir/'references/columns/flash.md', "Flash")
docstr2md(Flowsheet.__doc__, current_dir/'references/flowsheet.md', "Flowsheet")
docstr2md(Components.__doc__, current_dir/'references/components.md', "Components")
