from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    _requirements = f.readlines()

setup(
    name="numpyai",
    version="0.2.0",
    author="Aadya Chinubhai",
    author_email="aadyachinubhai@gmail.com",
    description="A library for extracting documentation and implementing a RAG system.",
    packages=find_packages(),
    install_requires=_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)