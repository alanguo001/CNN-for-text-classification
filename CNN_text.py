'''
@author Byron Wallace
A Keras implementation of CNNs for text classification. 

Credit for initial pass of implementation to: Cheng Guo (https://gist.github.com/entron).

References
--
Yoon Kim. "Convolutional Neural Networks for Sentence Classification". EMNLP 2014.
Ye Zhang and Byron Wallace. "A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification". http://arxiv.org/abs/1510.03820.
& also: http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
'''

from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import accuracy
from keras.preprocessing.text import text_to_word_sequence, Tokenizer

class TextCNN:

    def __init__(self, max_features=10000, maxlen=300, embedding_dims=200, 
                        filters=None, n_filters=100, dropout=0.0, wvs=None):
        '''
        parameters
        ---
        max_features: the upper bound to be placed on the vocabulary size.
        maxlen: the maximum length (in terms of tokens) of the instances/texts.
        embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors is provided (if wvs is not None).
        filters: list of unique filter sizes (i.e., ngram sizes) to use.
        n_filters: number of filters for each filter size.
        dropout: dropout rate for penultimate layer.
        wvs: pre-trained word vectors (e.g., trained Word2Vec model; assumes gensim object!)
        '''
        self.max_features = max_features  
        self.tokenizer = Tokenizer(nb_words=self.max_features)
        self.maxlen = maxlen  

        self.use_pretrained_embeddings = False 
        if wvs is None:
            self.embedding_dims = embedding_dims
        else:
            # note that these are only for initialization;
            # they will be tuned!
            self.use_pretrained_embeddings = True
            self.embedding_dims = wvs.vector_size
            self.word_embeddings = wvs

        if filters is None:
            self.ngram_filters = [3, 4, 5]
        else:
            self.ngram_filters = filters 

        self.nb_filter = n_filters 
        self.dropout = dropout

    def initialize_sequences_and_vocab(self, all_texts):
        ''' 
        This fits tokenizer and builds up input vectors (X) from the list 
        of texts in all_texts. Needs to be called before train!
        '''
        self.raw_texts = all_texts
        self.build_sequences()
        if self.use_pretrained_embeddings:
            self.init_word_vectors()

    def build_sequences(self):
        ''' Fits tokenizer and constructures sequences (i.e., X) '''
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token

        self.X = list(self.tokenizer.texts_to_sequences_generator(self.raw_texts))
        self.X = np.array(pad_sequences(self.X, maxlen=self.maxlen))

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

        self.init_vectors = np.vstack(self.init_vectors)

    
    def train(self, X_train, y_train, X_val=None, y_val=None,
                nb_epoch=5, batch_size=32, optimizer='rmsprop'):
        ''' 
        Accepts an X matrix (presumably some slice of self.X) and corresponding
        vector of labels. May want to revisit this. 

        X_val and y_val are to be used to validate during training. 
        '''
        self.build_model() # re-build model
        self.model.compile(loss={'output': 'binary_crossentropy'}, optimizer=optimizer)

        if X_val is not None:
            self.model.fit({'input': X_train, 'output': y_train},
                batch_size=batch_size, nb_epoch=nb_epoch,
                validation_data={'input': X_val, 'output': y_val})
        else: 
            print("no validation data provided!")
            self.model.fit({'input': X_train, 'output': y_train},
                batch_size=batch_size, nb_epoch=nb_epoch)

        

    def predictions(X_test, batch_size=32, binarize=False):
        raw_preds = np.array(self.model.predict({'input': X_test}, 
                                  batch_size=batch_size)['output'])
        if binarize:
          return np.round(raw_preds)
        return raw_preds


    def build_model(self):
        # again, credit to Cheng Guo
        self.model = Graph()
        self.model.add_input(name='input', input_shape=(self.maxlen,), dtype=int)
        init_embeddings = None 
        if self.use_pretrained_embeddings:
            init_embeddings = [self.init_vectors]

        self.model.add_node(Embedding(self.max_features, self.embedding_dims, 
                                input_length=self.maxlen, weights=init_embeddings), 
                                name='embedding', input='input')
        self.model.add_node(Dropout(0.), name='dropout_embedding', input='embedding')
        for n_gram in self.ngram_filters:
            self.model.add_node(Convolution1D(nb_filter=self.nb_filter,
                                         filter_length=n_gram,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=self.embedding_dims,
                                         input_length=self.maxlen),
                           name='conv_' + str(n_gram),
                           input='dropout_embedding')
            self.model.add_node(MaxPooling1D(pool_length=self.maxlen - n_gram + 1),
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

