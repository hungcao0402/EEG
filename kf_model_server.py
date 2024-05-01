from numpy import *
import numpy as np
import glob
import re
from pylab_sdk import *
from scipy.signal import *
import pandas as pd
import sys
from io import BytesIO
from typing import List, Dict
import base64
import io
from kserve import Model, ModelServer
from joblib import dump, load
from classif import updateMeta

# Define the custom model server class
class EEG(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.load()


    def load(self):
        self.model = load('/mnt/models/eeg_model.joblib')
        self.ready = True
        

    def preproc(self, filename, sig):
        def bandpass(sig,band,fs):
            B,A = butter(5, array(band)/(fs/2), btype='bandpass')
            return lfilter(B, A, sig, axis=0)

        reg = re.compile('\d+')
        
        freq = 200.0

        epoc_window = 1.3*freq

        X = []
        User = []
        idFeedBack = []
        Session = []
        Feedback = []
        Letter = []
        Word = []
        FeedbackTot = []
        LetterTot = []
        WordTot = []

        user,session = reg.findall(filename)
        # sig = np.array(pd.io.parsers.read_csv(f))

        EEG = sig[:,1:-2]
        EOG = sig[:,-2]
        Trigger = sig[:,-1]

        sigF = bandpass(EEG,[1.0,40.0],freq)

        idxFeedBack = np.where(Trigger==1)[0]    

        for fbkNum,idx in enumerate(idxFeedBack):
            X.append(sigF[idx:idx + int(epoc_window),:])
            User.append(int(user))
            idFeedBack.append('S' + user + '_Sess' + session + '_FB' + '%03d' % (fbkNum+1) )
            Session.append(int(session))
            Feedback.append(fbkNum)
            Letter.append(mod(fbkNum,5) + 1)
            Word.append(floor(fbkNum/5)+1)
            FeedbackTot.append(fbkNum + (int(session)-1)*60)
            WordTot.append(floor(fbkNum/5)+1 + (int(session)-1)*12)
        
        Meta = array([Session,Feedback,Letter,Word,FeedbackTot,WordTot]).transpose()
        
        Meta2 = pd.read_csv('preproc/metadata.csv')
        # currentUserSet = [True if val in set(User) else False for val in Meta2.subject]
        isLong = Meta2.loc[(Meta2['subject'] == 1) & (Meta2['session'] == 1), 'isLong'].values
        # isLong = Meta2.isLong[currentUserSet].values
        Meta = np.c_[Meta,isLong]

        info = array([idFeedBack,User])

        X = array(X).transpose((0,2,1)) # Ns,Ne,Nt

        return info, X, Meta

    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        csv_content = request["instances"][0]['content']
        filename = request["instances"][0]['filename']

        _input = base64.b64decode(csv_content)
        _input = io.BytesIO(_input)
        _input = np.array(pd.io.parsers.read_csv(_input))
        info, X, Meta = self.preproc(filename, _input)
        feedbackid, User = info 

        users_test = np.unique(User)
        ### predicting
        prob = []
        for ut in users_test:
            updateMeta(self.model,Meta[User==ut,...])
            # print('Running predict X_test', X_test[User_test==ut,...].shape, X_test[User_test==ut,...])
            prob.extend(self.model.predict(X[User==ut,...]))
        prob = np.array(prob)

        df = pd.DataFrame({'IdFeedBack':feedbackid,'Prediction':prob})
        return {"predictions": df.to_dict(orient='records')}

if __name__ == "__main__":
    model = EEG("eeg_model")
    ModelServer().start([model])
