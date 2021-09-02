[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
# Llyr
micromagnetic post processing library

transform a .out folder with .ovf files inside into a nice hdf5 (h5) file. h5 files let you store huge amounts of numerical data, and easily manipulate that data from NumPy. For example, you can slice into multi-terabyte datasets stored on disk, as if they were real NumPy arrays. Thousands of datasets can be stored in a single file, categorized and tagged however you want

## Installation

```
$ pip install llyr
```

## Usage

Creating
```python
>>> from llyr import Llyr
>>> job = Llyr("path/to/new/h5") # creating a blank h5 if it doesn't exist
```
Building from the *out* folder
```python
>>> job.make("path/to/out/folder")
```
Visualizations
```python
>>> job.p # list the datasets and attributes
>>> job.snapshot('dataset_name') # quick view 
```
Postprocessing
```python
>>> disp = job.disp("dataset_name") # calculating the dispersion
>>> fft = job.fft('dataset_name') # calculating the fft spectrum
```
