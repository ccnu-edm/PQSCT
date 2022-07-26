import pandas as pd
import codecs, gc
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
import re
from sklearn.preprocessing import MultiLabelBinarizer as MLB
import os
import json
import tensorflow as tf
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import sys
import io
def setup_io():
    sys.stdout = sys.__stdout__ = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', line_buffering=True)
    sys.stderr = sys.__stderr__ = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', line_buffering=True)
setup_io()
import importlib
importlib.reload(sys)

import warnings
warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

maxlen      = 128  
Batch_size  = 4   
# Epoch       = 10    
Epoch       = 20 
num_classes=68

pre_dir='pre_model/'
config_path =pre_dir+ 'bert_config.json'
dict_path =pre_dir+ 'vocab.txt'
checkpoint_path =pre_dir +'model.ckpt-100000'

proc_dir = r"dataset/"
train_data_path=os.path.join(proc_dir, "train.csv")
valid_data_path=os.path.join(proc_dir, "valid.csv")
test_data_path=os.path.join(proc_dir, "test.csv")
savepath='./'


def get_mlb():
    with open(os.path.join(proc_dir, "multi_class.txt"), "r",encoding='utf-8') as f:
        labels = [[_.strip()] for _ in f.readlines()]
    mlb = MLB()
    mlb.fit(labels)
    all_class = mlb.classes_.tolist()
    print("all_class=",len(all_class))
    return mlb


def load_data(filename):
    mlb=get_mlb()
    print("\nReading data ... \n")
    D=[]
    data = pd.read_csv(filename,sep='\t').astype(str)
    pattern = re.compile(r"\[|\]|\'|,")
    data['label'] = data['label'].apply(lambda src: re.sub(pattern, "", str(src)))
    data["label"] = data["label"].apply(lambda x:x.split())
    for data_row in data.iloc[:].itertuples():
        D.append(((data_row.item, data_row.resolve), mlb.transform([data_row.label])[0]))
    D = np.array(D)
    return D


def get_token_dict():
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') 
            else:
                R.append('[UNK]') 
        return R

tokenizer = OurTokenizer(get_token_dict())


def seq_padding(X, padding=0, ML=maxlen*2, label=False):
    if label == False:
        return np.array([ np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X])
    else:
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([ np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X])


class data_generator:
    def __init__(self, data, batch_size=Batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text1 = d[0][0][:maxlen]
                text2 = d[0][1][:maxlen]
                x1, x2 = tokenizer.encode(first=text1,second=text2)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1,padding=102)
                    X2 = seq_padding(X2,padding=1)
                    Y = seq_padding(Y, label=True)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []


class MaskedGlobalMaxPool1D(Layer):
    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPool1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs -= K.expand_dims((1.0 - mask) * 1e6, axis=-1)
        return K.max(inputs, axis=-2)


class MaskedGlobalAveragePooling1D(Layer):
    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)
        
    def call(self, x, mask=None):
        if mask is not None:
            mask = K.repeat(mask, x.shape[-1])
            mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(x, axis=1)


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


def build_bert(nclass,batch_size=Batch_size,max_len=maxlen):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None) 
    for l in bert_model.layers:
        l.trainable = True
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x = bert_model([x1_in, x2_in])
    c = Lambda(lambda x: x[:, 0])(x)  
    pool = MaskedGlobalMaxPool1D()(x)
    ave = MaskedGlobalAveragePooling1D()(x)
    x = Concatenate()([pool, ave, c])
    p = Dense(nclass, activation='linear')(x)
    model = Model([x1_in, x2_in], p)
    model.compile(#loss='binary_crossentropy',
                  loss=multilabel_categorical_crossentropy,
                  optimizer=Adam(1e-5),  
                  metrics=['accuracy', precision_m, recall_m, f1_m])
    print(model.summary())
    return model


def sigmoid_pre(mlb,model_pred):
    y_pred = []
    for i in range(len(model_pred)):
        indices = [j for j in range(len(model_pred[i])) if model_pred[i][j] > 0]
        y_pred.append([mlb.classes_.tolist()[index] for index in indices])
    return y_pred
    

def evaluate(valid_data,valid_D,model):
    valid_model_pred = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
    y_pred = mlb.transform(sigmoid_pre(mlb,valid_model_pred))
    y_true = mlb.transform(sigmoid_pre(mlb,valid_data[:, 1]))
    h = metrics.hamming_loss(y_true,y_pred)
    p = metrics.precision_score(y_true, y_pred, average='micro')
    r = metrics.recall_score(y_true, y_pred,average='micro')
    f1 = metrics.f1_score(y_true, y_pred,average='micro')
    f1_macro = metrics.f1_score(y_true, y_pred,average='macro')
    acc= metrics.accuracy_score(y_true, y_pred)
    return h,p,r,f1,f1_macro,acc


def write_t(h,p,r,f1,f1_macro,acc,valid_test):
    with open(savepath+"result.txt", "a",encoding='utf-8') as f:
        f.write('\r\n')
        f.write("valid or test ? {}\n".format(valid_test))
        f.write("hamming_loss is {}\n".format(h))
        f.write("precision_score is {}\n".format(p))
        f.write("recall_score is {}\n".format(r))
        f.write("f1_score is {}\n".format(f1))
        f.write("f1_macro_score is {}\n".format(f1_macro))
        f.write("accuracy_score is {}\n".format(acc))
        f.write('\r\n')


def run_kb():
    mlb=get_mlb()
    train_data=load_data(train_data_path)
    valid_data=load_data(valid_data_path)
    test_data=load_data(test_data_path)

    train_D = data_generator(train_data, shuffle=True)
    valid_D = data_generator(valid_data, shuffle=False)
    test_D = data_generator(test_data, shuffle=False)

    print('Loading model,Please wait!....')
    model = build_bert(num_classes)  
    print('loading model success! Training!....')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3,verbose=2,restore_best_weights=True)  
    plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='auto', factor=0.9, patience=2)  
    checkpoint = ModelCheckpoint(savepath+'bert_base.hdf5', monitor='val_accuracy', verbose=2,
                                 save_best_only=True, mode='auto', save_weights_only=True)  

    train_log = model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=Epoch,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        # callbacks=[early_stopping, plateau, checkpoint],
        # callbacks=[logs_loss]
        )
    #valid
    valid_h,valid_p,valid_r,valid_f,valid_f_macro,valid_acc=evaluate(valid_data,valid_D,model)
    write_t(valid_h,valid_p,valid_r,valid_f,valid_f_macro,valid_acc,'valid')
    #test
    test_h,test_p,test_r,test_f,test_f_macro,test_acc=evaluate(test_data,test_D,model)
    write_t(test_h,test_p,test_r,test_f,test_f_macro,test_acc,'test')
    model_path =savepath+'bert_'+str(Epoch)+'.h5'
    model.save(model_path)
    return test_h,test_p,test_r,test_f,test_f_macro,test_acc


if __name__ == '__main__':
    mlb=get_mlb()
    test_h,test_p,test_r,test_f,test_f_macro,test_acc=run_kb()
    print("test hamming_loss is {}".format(test_h))
    print("test precision_score is {}\n".format(test_p))
    print("test recall_score is {}".format(test_r))
    print("test f1_score is {}".format(test_f))
    print("test accuracy_score is {}".format(test_acc))