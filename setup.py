import setuptools

from smiley import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="drug-molecule-properties",
    version=__version__,
    author="Léo Grosjean",
    author_email="leo.grosjean@live.fr",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lgrosjean/drug-molecule-properties",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": ["servier=main:main"],
    },
)