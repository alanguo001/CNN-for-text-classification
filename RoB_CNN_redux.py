import csv
import sys
import os 
csv.field_size_limit(sys.maxsize)

import pandas as pd 
import numpy as np 

import gensim 
from gensim.models import Word2Vec

import CNN_text


def load_trained_w2v_model(path="/Users/byron/dev/Deep-PICO/PubMed-w2v.bin"):
    m = Word2Vec.load_word2vec_format(path, binary=True)
    return m


def read_RoB_data(path="RoB-data/train-Xy-Random-sequence-generation.txt", 
                    y_tuples=False, zero_one=True):
    ''' 
    Assumes data is in CSV with label as second entry.
    '''
    raw_texts, y = [], []
    with open(path) as input_file: 
        rows = csv.reader(input_file)
        for row in rows: 
            doc_text, lbl = row
            raw_texts.append(doc_text)
            cur_y = int(lbl)
            if y_tuples:
                if cur_y > 0:
                    y.append(np.array([0,1]))
                else: 
                    y.append(np.array([1,0]))
            else:
                if cur_y < 1:
                    if zero_one:
                        y.append(0)
                    else:
                        y.append(-1)
                    
                else:
                    y.append(1)

    return raw_texts, y 



def RoB_CNN(total_epochs=100):
    train_docs, y_train = read_RoB_data(path="RoB-data/train-Xy-Random-sequence-generation.txt", 
                                        y_tuples=False)
   
    test_docs, y_test = read_RoB_data(path="RoB-data/test-Xy-Random-sequence-generation.txt",
                                        y_tuples=False)

    
    train_docs = train_docs#[:500]
    y_train = y_train#[:500]

    wvs = load_trained_w2v_model()
    # preprocessor for texts

    # then the CNN
    p = CNN_text.Preprocessor(max_features=10000, maxlen=5000, wvs=wvs)
    all_docs = train_docs + test_docs
    
    print("preprocessing...")
    p.preprocess(all_docs)
    train_X = p.build_sequences(train_docs)
    test_X = p.build_sequences(test_docs)
    

    cnn = CNN_text.TextCNN(p, filters=[2,3,5], n_filters=100, dropout=0.0)

    epochs_per_iter = 10
    epochs_so_far = 0
    while epochs_so_far < total_epochs:
        cnn.train(train_X, y_train, nb_epochs=epochs_per_iter)#, X_val=test_X, y_val=y_test)
        epochs_so_far += epochs_per_iter
        
        y_hat = cnn.predict(test_X)
        import pdb; pdb.set_trace()
        print("acc")
    #cnn.initialize_sequences_and_vocab(all_docs)
    #cnn.train(X_train, y_train, X_val=None, y_val=None


# note that on TACC you need:
#    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/intel14/hdf5/1.8.12/x86_64/lib/
if __name__ == '__main__':
    RoB_CNN()
