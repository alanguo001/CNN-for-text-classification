'''
@author Byron Wallace
A Keras implementation of our "rationale augmented CNN" (https://arxiv.org/abs/1605.04469).

Credit for initial pass of implementation to: Cheng Guo (https://gist.github.com/entron).

References
--
Ye Zhang, Iain J. Marshall and Byron C. Wallace. "Rationale-Augmented Convolutional Neural Networks for Text Classification". http://arxiv.org/abs/1605.04469
Yoon Kim. "Convolutional Neural Networks for Sentence Classification". EMNLP 2014.
Ye Zhang and Byron Wallace. "A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification". http://arxiv.org/abs/1510.03820.
& also: http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
'''

from __future__ import print_function
import pdb
import sys
import random
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np

#import nltk # for sentence splitting 
#from nltk.tokenize import sent_tokenize

from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import accuracy
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint

##
from keras.layers import Input, Embedding, Dense, merge
from keras.models import Model


class RationaleCNN:

    def __init__(self, preprocessor, filters=None, n_filters=100, dropout=0.0):
        '''
        parameters
        ---
        preprocessor: an instance of the Preprocessor class, defined below
        '''
        self.preprocessor = preprocessor

        if filters is None:
            self.ngram_filters = [3, 4, 5]
        else:
            self.ngram_filters = filters 

        self.nb_filter = n_filters 
        self.dropout = dropout

        #self.build_model() # build model
        #self.train_sentence_model()

    @staticmethod
    def balanced_sample(X, y):
        _, pos_rationale_indices = np.where([y[:,0] > 0]) 
        _, neg_rationale_indices = np.where([y[:,1] > 0]) 
        _, non_rationale_indices = np.where([y[:,2] > 0]) 

        # sample a number of non-rationales equal to the total
        # number of pos/neg rationales. 
        m = pos_rationale_indices.shape[0] + neg_rationale_indices.shape[0]
        sampled_non_rationale_indices = np.array(random.sample(non_rationale_indices, m))

        train_indices = np.concatenate([pos_rationale_indices, neg_rationale_indices, sampled_non_rationale_indices])
        np.random.shuffle(train_indices) # why not
        return X[train_indices,:], y[train_indices]

    # r_CNN.sentence_model.predict(X[:10], batch_size=128)
    def train_sentence_model(self, train_documents, nb_epoch=5, batch_size=128, optimizer='adam'):
        # assumes sentence sequences have been generated!
        assert(train_documents[0].sentence_sequences is not None)

        X, y= [], []
        # flatten sentences/sentence labels
        for d in train_documents:
            X.extend(d.sentence_sequences)
            y.extend(d.sentences_y)

        # @TODO sub-sample magic?
        X, y = np.asarray(X), np.asarray(y)
        
        # downsample
        X_ds, y_ds = RationaleCNN.balanced_sample(X, y)

        #self.train(X[:1000], y[:1000])
        self.train(X_ds, y_ds)


    def train(self, X_train, y_train, X_val=None, y_val=None,
                nb_epoch=5, batch_size=32, optimizer='adam'):
        ''' 
        Accepts an X matrix (presumably some slice of self.X) and corresponding
        vector of labels. May want to revisit this. 

        X_val and y_val are to be used to validate during training. 
        '''


        checkpointer = ModelCheckpoint(filepath="weights.hdf5", 
                                       verbose=1, 
                                       save_best_only=(X_val is not None))

        if X_val is not None:
            self.sentence_model.fit({'input': X_train, 'output': y_train},
                batch_size=batch_size, nb_epoch=nb_epoch,
                validation_data={'input': X_val, 'output': y_val},
                verbose=2, callbacks=[checkpointer])
        else: 
            print("no validation data provided!")
            #self.sentence_model.fit({'input': X_train, 'output': y_train},
            #    batch_size=batch_size, nb_epoch=nb_epoch, 
            #    verbose=2, callbacks=[checkpointer])
            self.sentence_model.fit(X_train, y_train,
                batch_size=batch_size, nb_epoch=nb_epoch, 
                verbose=2, callbacks=[checkpointer])
        

    '''
    def predict(self, X_test, batch_size=32, binarize=False):
        raw_preds = self.model.predict({'input': X_test}, batch_size=batch_size)['output']

        #np.array(self.model.predict({'input': X_test}, 
                    #              batch_size=batch_size)['output'])
        if binarize:
          return np.round(raw_preds)
        return raw_preds
    '''


    def build_sentence_model(self):
        tokens_input = Input(name='input', shape=(self.preprocessor.maxlen,), dtype='int32')
        x = Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                      input_length=self.preprocessor.maxlen, 
                      weights=self.preprocessor.init_vectors)(tokens_input)
        
        x = Dropout(0.1)(x)

        convolutions = []
        for n_gram in self.ngram_filters:
            cur_conv = Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=n_gram,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=self.preprocessor.embedding_dims,
                                         input_length=self.preprocessor.maxlen)(x)
            # pool
            one_max = MaxPooling1D(pool_length=self.preprocessor.maxlen - n_gram + 1)(cur_conv)
            flattened = Flatten()(one_max)
            convolutions.append(flattened)

        penultimate_layer = merge(convolutions)
        output = Dense(3, activation="softmax")(penultimate_layer)

        self.sentence_model = Model(input=tokens_input, output=output)
        print("model built")
        print(self.sentence_model.summary())
        self.sentence_model.compile(loss='categorical_crossentropy', optimizer="adam")

        return self.sentence_model 


    def build_doc_model(self):
        # again, credit to Cheng Guo
        self.model = Graph()
        self.model.add_input(name='input', input_shape=(self.preprocessor.maxlen,), dtype=int)

        self.model.add_node(Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                                input_length=self.preprocessor.maxlen, weights=self.preprocessor.init_vectors), 
                                name='embedding', input='input')
        self.model.add_node(Dropout(0.), name='dropout_embedding', input='embedding')
        for n_gram in self.ngram_filters:
            self.model.add_node(Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=n_gram,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=self.preprocessor.embedding_dims,
                                         input_length=self.preprocessor.maxlen),
                           name='conv_' + str(n_gram),
                           input='dropout_embedding')
            self.model.add_node(MaxPooling1D(pool_length=self.preprocessor.maxlen - n_gram + 1),
                           name='maxpool_' + str(n_gram),
                           input='conv_' + str(n_gram))
            self.model.add_node(Flatten(),
                           name='flat_' + str(n_gram),
                           input='maxpool_' + str(n_gram))
        self.model.add_node(Dropout(self.dropout), name='dropout', inputs=['flat_' + str(n) for n in self.ngram_filters])
        self.model.add_node(Dense(1, input_dim=self.nb_filter * len(self.ngram_filters)), 
                                  name='dense', input='dropout')
        self.model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
        self.model.add_output(name='output', input='sigmoid')
        print("model built")
        print(self.model.summary())
        self.model.compile(loss={'output': 'binary_crossentropy'}, 
                                optimizer="adam")#optimizer)


class Document:
    def __init__(self, doc_id, sentences, doc_label=None, sentences_labels=None):
        self.doc_id = doc_id
        self.doc_y = doc_label

        self.sentences = sentences
        self.sentence_sequences = None

        self.sentences_y = sentences_labels

        self.sentence_idx = 0
        self.n = len(self.sentences)


    def __len__(self):
        return self.n 

    def generate_sequences(self, p):
        ''' 
        p is a preprocessor that has been instantiated
        elsewhere! this will be used to map sentences to 
        integer sequences here.
        '''
        self.sentence_sequences = p.build_sequences(self.sentences)



    '''
    def __iter__(self):
        return self 

    def next(self):
        if self.sentence_idx < self.n:
            cur_sentence_idx = self.sentence_idx 
            self.sentence_idx += 1

            return self.sentences[cur_sentence_idx]
        else:
            raise StopIteration()

    '''


class Preprocessor:
    def __init__(self, max_features, maxlen, embedding_dims=200, wvs=None):
        '''
        max_features: the upper bound to be placed on the vocabulary size.
        maxlen: the maximum length (in terms of tokens) of the instances/texts.
        embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors is provided (if wvs is not None).
        '''

        self.max_features = max_features  
        self.tokenizer = Tokenizer(nb_words=self.max_features)
        self.maxlen = maxlen  

        self.use_pretrained_embeddings = False 
        self.init_vectors = None 
        if wvs is None:
            self.embedding_dims = embedding_dims
        else:
            # note that these are only for initialization;
            # they will be tuned!
            self.use_pretrained_embeddings = True
            self.embedding_dims = wvs.vector_size
            self.word_embeddings = wvs


    def preprocess(self, all_docs):
        ''' 
        This fits tokenizer and builds up input vectors (X) from the list 
        of texts in all_texts. Needs to be called before train!
        '''
        self.raw_texts = all_docs
        #self.build_sequences()
        self.fit_tokenizer()
        if self.use_pretrained_embeddings:
            self.init_word_vectors()


    def fit_tokenizer(self):
        ''' Fits tokenizer to all raw texts; remembers indices->words mappings. '''
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token

    def build_sequences(self, texts):
        X = list(self.tokenizer.texts_to_sequences_generator(texts))
        X = np.array(pad_sequences(X, maxlen=self.maxlen))
        return X

    def init_word_vectors(self):
        ''' 
        Initialize word vectors.
        '''
        self.init_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.max_features:
                try:
                    self.init_vectors.append(self.word_embeddings[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.embedding_dims)*-2 + 1

                    self.init_vectors.append(unknown_words_to_vecs[t])

        # note that we make this a singleton list because that's
        # what Keras wants. 
        self.init_vectors = [np.vstack(self.init_vectors)]

