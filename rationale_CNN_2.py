'''
@authors Byron Wallace, Edward Banner, Ye Zhang

A Keras implementation of our "rationale augmented CNN" (https://arxiv.org/abs/1605.04469). Please note that
the model was originally implemented in Theano -- this version is a work in progress.

Credit for initial pass of basic CNN implementation to: Cheng Guo (https://gist.github.com/entron).

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

from keras import backend as K 
from keras.models import Graph, Model, Sequential
from keras.preprocessing import sequence
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Reshape, Permute, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D
from keras.utils.np_utils import accuracy
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.callbacks import ModelCheckpoint



class RationaleCNN:

    def __init__(self, preprocessor, filters=None, n_filters=32, dropout=0.0):
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

        self.n_filters = n_filters 
        self.dropout = dropout
        self.sentence_model_trained = False 

    @staticmethod
    def weighted_sum(X):
        # @TODO.. add sentence preds!
        return K.sum(X, axis=0) 

    @staticmethod
    def weighted_sum_output_shape(input_shape):
        # expects something like (None, max_doc_len, num_features) 
        # returns (1 x num_features)
        shape = list(input_shape)
        return tuple((1, shape[-1]))

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

    def get_conv_layers_from_sentence_model():
        layers_to_weights = {}
        for ngram in self.ngram_filters:
            layer_name = "conv_" + str(ngram)
            cur_conv_layer = self.sentence_model.get_layer(layer_name)
            weights, biases = cur_conv_layer.get_weights()

            # here it gets tricky because we need
            # so, e.g., (32 x 200 x 3 x 1) -> (32 x 3 x 200 x 1)
            # we do this because reshape by default iterates over
            # the last dimension fastest
            # swapped = np.swapaxes(X, 1, 2)
            # Xp = swapped.reshape(32, 1, 1, 600)

    def build_doc_model(self):
        #assert self.sentence_model_trained

        # input dim is (max_doc_len x max_sent_len) -- eliding the batch size
        tokens_input = Input(name='input', 
                            shape=(self.preprocessor.max_doc_len, self.preprocessor.max_sent_len), 
                            dtype='int32')
        # flatten; create a very wide matrix to hand to embedding layer
        tokens_reshaped = Reshape([self.preprocessor.max_doc_len*self.preprocessor.max_sent_len])(tokens_input)
        # embed the tokens; output will be (p.max_doc_len*p.max_sent_len x embedding_dims)
        # here we should initialize with weights from sentence model embedding layer!


        ### 
        # getting weights for initialization
        x = Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                        weights=self.sentence_model.get_layer("embedding").get_weights(),
                        #weights=self.preprocessor.init_vectors, 
                        name="embedding")(tokens_reshaped)

        # reshape to preserve document structure; each doc will now be a
        # a row in this matrix
        x = Reshape((1, self.preprocessor.max_doc_len, 
                     self.preprocessor.max_sent_len*self.preprocessor.embedding_dims), 
                     name="reshape")(x)

        x = Dropout(0.1, name="dropout")(x)

        convolutions = []
        for n_gram in self.ngram_filters:


            #import pdb; pdb.set_trace()

            ### here is where we pull out weights
            layer_name = "conv_" + str(n_gram)
            cur_conv_layer = self.sentence_model.get_layer(layer_name)
            weights, biases = cur_conv_layer.get_weights()
            # here it gets a bit tricky; we need dims 
            #       (nb_filters x 1 x 1 x (n_gram*embedding_dim))
            # for 2d conv; our 1d conv model, though, will have
            #       (nb_filters x embedding_dim x n_gram x 1)
            # need to reshape this. but first need to swap around
            # axes due to how reshape works (it iterates over last 
            # dimension first). in particular, e.g.,:
            #       (32 x 200 x 3 x 1) -> (32 x 3 x 200 x 1)
            # swapped = np.swapaxes(X, 1, 2)
            swapped_weights = np.swapaxes(weights, 1, 2)
            init_weights = swapped_weights.reshape(self.n_filters, 
                            1, 1, n_gram*self.preprocessor.embedding_dims)

            cur_conv = Convolution2D(self.n_filters, 1, 
                                     n_gram*self.preprocessor.embedding_dims, 
                                     subsample=(1, self.preprocessor.embedding_dims),
                                     name="conv2d_"+str(n_gram),
                                     weights=[init_weights, biases])(x)

            # this output (n_filters x max_doc_len x 1)
            one_max = MaxPooling2D(pool_size=(1, self.preprocessor.max_sent_len - n_gram + 1), 
                                   name="pooling_"+str(n_gram))(cur_conv)

            # flip around, to get (1 x max_doc_len x n_filters)
            permuted = Permute((3,2,1), name="permute_"+str(n_gram)) (one_max)
            
            # drop extra dimension
            r = Reshape((self.preprocessor.max_doc_len, self.n_filters), 
                            name="conv_"+str(n_gram))(permuted)
            
            convolutions.append(r)

        # merge the filter size convolutions
        r = merge(convolutions, name="sentence_vectors")

        # now we take a weighted sum of the sentence vectors
        # to induce a document representation
        x_doc = Lambda(RationaleCNN.weighted_sum, 
                        output_shape=RationaleCNN.weighted_sum_output_shape, 
                        name="weighted_doc_vector")(r)

        # finally, the sigmoid layer for classification
        y_hat = Dense(1, activation="softmax", name="document_prediction")(x_doc)
        model = Model(input=tokens_input, output=x_doc)
        return model 
        

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
            self.sentence_model.fit(X_train, y_train,
                batch_size=batch_size, nb_epoch=nb_epoch,
                validation_data=(X_val, y_val),
                verbose=2, callbacks=[checkpointer])
        else: 
            # no validation provided
            self.sentence_model.fit(X_train, y_train,
                batch_size=batch_size, nb_epoch=nb_epoch, 
                verbose=2, callbacks=[checkpointer])


    '''
    def build_sentence_model(self):

        # input dim is (max_doc_len x max_sent_len) -- eliding the batch size
        tokens_input = Input(name='input', 
                            shape=(self.preprocessor.max_sent_len,), 
                            dtype='int32')

        # embed the tokens; output will be (p.max_doc_len*p.max_sent_len x embedding_dims)
        x = Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                        weights=self.preprocessor.init_vectors, name="embedding")(tokens_input)

        x = Dropout(0.1, name="dropout")(x)

        convolutions = []
        for n_gram in self.ngram_filters:
            cur_conv = Convolution2D(self.n_filters, 1, 
                                     n_gram*self.preprocessor.embedding_dims, 
                                     subsample=(1, self.preprocessor.embedding_dims),
                                     name="conv2d_"+str(n_gram))(x)

            # this output (n_filters x max_doc_len x 1)
            one_max = MaxPooling2D(pool_size=(1, self.preprocessor.max_sent_len - n_gram + 1), 
                                   name="pooling_"+str(n_gram))(cur_conv)

            # flip around, to get (1 x max_doc_len x n_filters)
            permuted = Permute((3,2,1), name="permute_"+str(n_gram)) (one_max)
            
            # drop extra dimension
            r = Reshape((self.preprocessor.max_doc_len, self.n_filters), 
                            name="conv_"+str(n_gram))(permuted)
            
            convolutions.append(r)

        # merge the filter size convolutions
        r = merge(convolutions, name="sentence_vectors")

        # now the classification layer...
    '''
   
    def build_sentence_model(self):
        ''' 
        Build the *sentence* level model, which operates over, erm, sentences. 
        The task is to predict which sentences are pos/neg rationales.
        '''
        tokens_input = Input(name='input', shape=(self.preprocessor.max_sent_len,), dtype='int32')
        x = Embedding(self.preprocessor.max_features, self.preprocessor.embedding_dims, 
                      name="embedding",
                      input_length=self.preprocessor.max_sent_len, 
                      weights=self.preprocessor.init_vectors)(tokens_input)
        
        x = Dropout(0.1)(x)

        convolutions = []
        for n_gram in self.ngram_filters:
            cur_conv = Convolution1D(name="conv_" + str(n_gram), 
                                         nb_filter=self.n_filters,
                                         filter_length=n_gram,
                                         border_mode='valid',
                                         activation='relu',
                                         subsample_length=1,
                                         input_dim=self.preprocessor.embedding_dims,
                                         input_length=self.preprocessor.max_sent_len)(x)
            # pool
            one_max = MaxPooling1D(pool_length=self.preprocessor.max_sent_len - n_gram + 1)(cur_conv)
            flattened = Flatten()(one_max)
            convolutions.append(flattened)

        sentence_vector = merge(convolutions, name="sentence_vector") # hang on to this layer!
        output = Dense(3, activation="softmax", name="sentence_prediction")(sentence_vector)

        self.sentence_model = Model(input=tokens_input, output=output)
        print("model built")
        print(self.sentence_model.summary())
        self.sentence_model.compile(loss='categorical_crossentropy', optimizer="adam")

        self.sentence_embedding_dim = self.sentence_model.layers[-2].output_shape[1]

        return self.sentence_model 


    def train_sentence_model(self, train_documents, nb_epoch=5, downsample=True, 
                                    batch_size=128, optimizer='adam'):
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
        if downsample:
            X, y = RationaleCNN.balanced_sample(X, y)

        #self.train(X[:1000], y[:1000])
        self.train(X, y)

        self.sentence_model_trained = True


class Preprocessor:
    def __init__(self, max_features, max_sent_len, embedding_dims=200, wvs=None, max_doc_len=500):
        '''
        max_features: the upper bound to be placed on the vocabulary size.
        max_sent_len: the maximum length (in terms of tokens) of the instances/texts.
        embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors is provided (if wvs is not None).
        '''

        self.max_features = max_features  
        self.tokenizer = Tokenizer(nb_words=self.max_features)
        self.max_sent_len = max_sent_len  # the max sentence length! @TODO rename; this is confusing. 
        self.max_doc_len = max_doc_len # w.r.t. number of sentences!

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
        X = np.array(pad_sequences(X, maxlen=self.max_sent_len))
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
