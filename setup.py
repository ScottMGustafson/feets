"""
setup script
"""

from io import open
from os import path
import versioneer
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def get_install_requires():
    with open("requirements.txt", "r") as f:
        lst = [x.strip() for x in f.read().split("\n") if not x.startswith("#")]
    return [x for x in lst if len(x) > 0]


def get_main_author():
    with open("AUTHORS.md", "r") as f:
        authors = f.readlines()
    main_author, main_email = authors[0].split(" : ")
    return main_author, main_email


def main():
    main_author, main_email = get_main_author()
    setup(
        name="feets",  # Required
        author=main_author,
        author_email=main_email,
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.7",
        ],
        packages=find_packages(exclude=["tests", "notebooks"]),  # Required
        python_requires=">=3.6",
        install_requires=get_install_requires(),  # Optional
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )


if __name__ == "__main__":
    main()
