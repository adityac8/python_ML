# python_ML
Python Machine Learning
Setting up floyd

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>floyd init mlp

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>cd nfiles

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>floyd data init mydata

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>floyd data upload

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>cd ..

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>floyd run --data <DATA_KEY>:nfiles --env theano-0.9:py2 "python mlp.py"

(D:\tenso\Anaconda2) D:\aditya\floyd\kaggle>floyd logs <RUN_ID>



x_data = np.load('/nfiles/fileone.npy')
y_data = np.load('/nfiles/filetwo.npy')
