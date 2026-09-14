"""
setup.py — packaging script for llm_finetune.

Note: this project can alternatively be packaged using pyproject.toml
(the modern PEP 517/518 standard). This setup.py is provided for
compatibility with tools/workflows that still expect one. You do not
need both — pick one as your source of truth.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="llm_finetune",
    version="0.1.0",
    description="QLoRA fine-tuning of stabilityai/stablelm-zephyr-3b on the Guanaco instruction dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joseph Akpe Unimke",
    url="",  # add your repo URL here
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-finetune-train=llm_finetune.train:main",
            "llm-finetune-infer=llm_finetune.inference:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

"Setup for standard details of build and implementation principles"