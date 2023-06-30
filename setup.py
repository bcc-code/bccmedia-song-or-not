import os
import platform
import sys

import pkg_resources
from setuptools import find_packages, setup

requirements = []
if sys.platform.startswith("linux") and platform.machine() == "x86_64":
    requirements.append("triton==2.0.0")

setup(
    name="bccmedia-song-or-not",
    py_modules=["inference"],
    version="1.0.0",
    description="Robust Speech Recognition via Large-Scale Weak Supervision",
    long_description=open("readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="readme.md",
    python_requires=">=3.8",
    author="BCC Code",
    url="https://github.com/bcc-code/bccmedia-song-or-not",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements
                     + [
                         str(r)
                         for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
                     ],
    include_package_data=True,
)