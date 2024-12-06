import logging

from setuptools import find_packages, setup
from setuptools.command.install import install

# Setup logging
logging.basicConfig(level=logging.INFO)

extras_require = {
    # Tasks
    "fairseq": ["fairseq @ git+https://github.com/ritsukkiii/fairseq.git",
                "transformers>=4.44.2",
                "accelerate",
                "tokenizers",
                "datasets",
                "sentencepiece!=0.99.1",
                "protobuf",
                "sacrebleu",
                "py7zr",
                "spacy",
                "torch>=1.13.1",
                "sentence-transformers==2.5.1",
                "tensorboardX",],
    "llama-factory": ["llama-factory[torch, metrics] @ git+https://github.com/hiyouga/LLaMA-Factory.git"],
}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="proxylm",
    version="1.0.0",
    author="David Anugraha",
    author_email="david.anugraha@gmail.com",
    description="ProxyLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/davidanugraha/proxylm",
    project_urls={
        "Bug Tracker": "https://github.com/davidanugraha/proxylm/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "numpy>=1.26.4",
        "pandas",
        "matplotlib",
        "xgboost>=2.1.1",
        "lightgbm>=4.5.0",
        "scikit-learn>=1.5.2",
        "scikit-optimize",
        "scipy",
        "pyyaml",
    ],
    package_dir={"": "src"},
    packages = find_packages("src"),
    entry_points={
        "console_scripts": [
            "proxylm-cli=proxy_regressor.main:main",
        ],
    },
    extras_require=extras_require,
    python_requires=">=3.10",
)
