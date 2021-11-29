[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
# Llyr
micromagnetic post processing library

Wrapper around a `zarr.hierarchy.Group` to implement convenience functions that work with the ouput of a modified mumax3.

## Installation

```
$ pip install llyr
```

## Usage

#### Creating
```python
import llyr
job = llyr.open("path/to/out/folder")
# or through any remote protocol 
job = llyr.open("ssh://username@remote.com:/home/username/data1.zarr/")
```
#### Visualizations

```python
job.p # list a dataset tree
job.snapshot('dataset_name') # quick view 
```
#### Accessing data
```python
arr = job.dataset_name[[0,25],...,2] # Numpy fancy indexing works too
```
