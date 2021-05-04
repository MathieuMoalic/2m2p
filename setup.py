from setuptools import find_packages, setup

setup(
    name="lyr",
    version="0.1",
    description="micromagnetic post processing library",
    author="Mathieu Moalic",
    author_email="matmoa@pm.me",
    platforms=["any"],
    license="GPL-3.0",
    url="https://github.com/MathieuMoalic/lyr",
    packages=find_packages(),
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
)
