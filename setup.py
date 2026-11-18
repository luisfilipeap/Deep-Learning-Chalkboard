from setuptools import setup, find_packages

setup(
    name="dl-chalkboard",
    version="0.1.0",
    author="Luis",
    description=(
        "DL ChalkBoard: deep learning from scratch, NumPy only, on any CPU "
        "(a tensor-valued successor to micrograd)."
    ),
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.8",
)
