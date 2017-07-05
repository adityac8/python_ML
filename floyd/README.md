# python_ML
Python Machine Learning
Setting up floyd

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>floyd init mlp

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>cd nfiles

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle\nfiles>floyd data init mydata

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle\nfiles>floyd data upload

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle\nfiles>cd ..

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>cd codes_folder

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle\codes_folder>floyd run --data <DATA_KEY>:nfiles --env theano-0.9:py2 "python mlp.py"

NOTE: Adding the codes in a separate folder prevents re-loading of data in the memory of floyd server. Floyd server synchronizes the WHOLE FOLDER in which the code is located when you run the code using the above command, which is actually a waste of (upload) time and a problem of data redundancy as you have already uploaded the data on the server. 

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle\codes_folder>floyd logs <RUN_ID>

x_data = np.load('/nfiles/fileone.npy')
y_data = np.load('/nfiles/filetwo.npy')

DEBUGGING NOTES: 

PROBLEM: If you are using Anaconda prompt and you come across the following error after running the command: "pip install -U floyd-cli":
scandir could not be installed

SOLUTION: conda install -c conda-forge scandir=1.5
