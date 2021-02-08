# Authors: Léo Grosjean <leo.grosjean@live.fr>
# License: GPL3

import setuptools

from smiley import __version__

with open("docs/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smiley",
    version=__version__,
    author="Léo Grosjean",
    author_email="leo.grosjean@live.fr",
    description="A small package to predict molecule properties based en Smile representation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lgrosjean/drug-molecule-properties",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": ["servier=smiley.main:main"],
    },
)