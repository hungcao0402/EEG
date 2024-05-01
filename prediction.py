from time import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
import yaml
import sys
from classif import updateMeta, baggingIterator
from multiprocessing import Pool
from functools import partial
import os
from joblib import dump, load

data_path = sys.argv[2]

def from_yaml_to_func(method,params):
    prm = dict()
    if params!=None:
        for key,val in params.items():
            prm[key] = eval(str(val)) 
    return eval(method)(**prm)


def BaggingFunc(bag,Labels,X,Meta,User,X_test,Meta_test,User_test):
    bagUsers = np.array([True if u in set(bag) else False for u in User])
    train_index = (~bagUsers)
    updateMeta(clf,Meta[train_index])
    clf.fit(X[train_index,:,:],Labels[train_index])
    dump(clf, os.path.join(data_path, 'eeg_model.joblib'))

    ### predicting
    prob = []
    for ut in users_test:
        updateMeta(clf,Meta_test[User_test==ut,...])
        # print('Running predict X_test', X_test[User_test==ut,...].shape, X_test[User_test==ut,...])
        prob.extend(clf.predict(X_test[User_test==ut,...]))
    prob = np.array(prob)
    
    return prob

# load parameters file
yml = yaml.load(open(sys.argv[1]), Loader=yaml.Loader)
# imports 
for pkg, functions in yml['imports'].items():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# parse pipe function from parameters
pipe = []
for item in yml['pipeline']:
    for method,params in item.items():
        pipe.append(from_yaml_to_func(method,params))

# create pipeline
clf = make_pipeline(*pipe)

opts=yml['MetaPipeline']
if opts is None:
    opts = {}

cores = yml['Submission']['cores']

# load data
X = np.load(os.path.join(data_path, './preproc/epochs.npy'))
Labels,User = np.load(os.path.join(data_path, './preproc/infos.npy'))
users = np.unique(User)
Meta = np.load(os.path.join(data_path, './preproc/meta_leak.npy')) if 'leak' in opts.keys() else np.load(os.path.join(data_path, './preproc/meta.npy'))

X_test = np.load(os.path.join(data_path, './preproc/test_epochs.npy'))
feedbackid,User_test = np.load(os.path.join(data_path, './preproc/test_infos.npy'))
User_test = np.array([int(u) for u in User_test])
users_test = np.unique(User_test)
Meta_test = np.load(os.path.join(data_path, './preproc/test_meta_leak.npy')) if 'leak' in opts.keys() else np.load(os.path.join(data_path, './preproc/test_meta.npy'))

### training
np.random.seed(5)
allProb = 0 

if 'bagging' in opts.keys():
    bagging = baggingIterator(opts, users)
else:
    bagging = [[-1]]

t = time()
pBaggingFunc = partial(BaggingFunc,Labels=Labels,X=X,Meta=Meta,User=User,X_test=X_test,Meta_test=Meta_test,User_test=User_test)
pool = Pool(processes = cores)
print(f'Running pool map in {cores} cores...')
allProb = pool.map(pBaggingFunc,bagging,chunksize=1)
allProb = np.vstack(allProb)
allProb = np.mean(allProb,axis=0)

if 'leak' in opts.keys():
    allProb += opts['leak']['coeff']*(1-Meta_test[:,-1])

print ("Done in " + str(time()-t) + " second")

submission = yml['Submission']['path']
df = pd.DataFrame({'IdFeedBack':feedbackid,'Prediction':allProb})
df.to_csv(os.path.join(data_path, submission),index=False)
print(f'Saved file in {submission}. Done!')