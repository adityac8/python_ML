# python_ML
Python Machine Learning
## Python Machine Learning Setting up floyd

Setting up floyd for using on Windows

## Installation

```
$ pip install -U floyd-cli
```

## Login

```
$ floyd login
```

Copy and paste your authentication token in your terminal

## Run project

**Make a project folder**

```
$ mkdir myproject
```

**Change to project folder**

```
$ cd myproject
```

**Make a folder for data**

```
$ mkdir mydata
```

**Make a folder for code**

```
$ mkdir mycode
```

Paste all your data in ```mydata```

Paste all your code in ```mycode```

**Change to data directory**

```
$ cd mydata
```

# Initialise data

```
$ floyd data init sent-data
```

Then you can upload your dataset to Floyd.

```
$ floyd data upload
```

Floyd will generate a data id for the uploaded dataset. This uploaded dataset can be used in your future experiments, if needed, using this data id.

**Output**

```
Creating data source. Uploading files ...
DATA ID                 NAME                    VERSION
----------------------  --------------------  ---------
GY3QRFFUA8KpbnqvroTPPW  alice/sent-data:1            1
```

**Change to main directory**
```
$ cd ..
```

**Change to code directory**
```
$ cd mycode
```

**Run the code**
```
$ floyd run --data <DATA_KEY>:<MOUNT> --env theano-0.9:py2 "python filename.py"
```

**Example code**
```
$ floyd run --data GY3QRFFUA8KpbnqvroTPPW:mydata --env theano-0.9:py2 "python mlp.py"
```

**Check the status of code**
```
$ floyd logs <RUN_ID>
```

# Some common problems:

* Should I put my code and data in separate folders

NOTE: Adding the codes in a separate folder prevents re-loading of data in the memory of floyd server.
Floyd server synchronizes the WHOLE FOLDER in which the code is located when you run the code using the above command, which is actually a waste of (upload) time and a problem of data redundancy as you have already uploaded the data on the server.


* How to link data in my script
```
x_data = np.load('/nfiles/fileone.npy')
y_data = np.load('/nfiles/filetwo.npy')
```

Here, nfiles is my MOUNT point.

* If you are using Anaconda prompt and you come across the following error after running the command: "pip install -U floyd-cli": scandir could not be installed
```
$ conda install -c conda-forge scandir=1.5
```

*Contributors:*
* Aditya Arora
* Piyush Jha
* Akshita Gupta
