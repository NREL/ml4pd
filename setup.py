from setuptools import setup, find_packages
import pathlib
from datetime import date


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
today = date.today().strftime(r"%Y.%m.%d")

requirements = (here / "requirements.txt").read_text(encoding="utf-8").split("\n")
test = [
    "seaborn",
    "ipykernel",
    "ipython",
    "pytest",
    "black",
    "pylint",
    "tqdm",
    "mkdocs-material",
    "mkdocs-jupyter",
]

setup(
    name="ml4pd",
    version=today,
    maintainer="Hien Vo",
    maintainer_email="nolan.wilson@nrel.gov",
    description="ML4PD - an open-source libray for building Aspen-like process models via machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=["Programming Language :: Python :: 3.7", "Programming Language :: Python :: 3.8"],
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    extras_require={"test": test},
)
