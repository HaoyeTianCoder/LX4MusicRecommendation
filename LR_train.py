import json
import pandas as pd
import numpy as np
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from config import *
import random
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import math

def gen_x_y():
    with open(song_popularity,'r',encoding='utf-8') as f1:
        popularity_dict = json.load(f1)
    with open(song_artist,'r',encoding='utf-8') as f2:
        song_artist_dict = json.load(f2)
    with open(user_like,'r',encoding='utf-8') as f3:
        user_like_dict = json.load(f3)
    with open(song_year,'r',encoding='utf-8') as f4:
        song_year_dict = json.load(f4)
    with open(song_track,'r',encoding='utf-8') as f5:
        song_track_dict = json.load(f5)
    with open(track_list,'r',encoding='utf-8') as f6:
        track_list_dict = json.load(f6)
    with open(user_like_genre,'r',encoding='utf-8') as f7:
        user_like_genre_dict = json.load(f7)
    with open(song_genre,'r',encoding='utf-8') as f8:
        song_genre_dict = json.load(f8)

    year_average = all = 0.0
    for k,v in song_year_dict.items():
        if v != None:
            year_average += float(v)
            all += 1
    year_average /= all

    song_hottness_average = all = 0.0
    for k,v in track_list_dict.items():
        if v[-1] != None and not math.isnan(v[-1]):
            song_hottness_average += float(v[-1])
            all += 1
    song_hottness_average /= all
    with open(train_v, 'r', encoding='utf-8') as f:
        X = []
        Y = []
        i = 0
        for line in f:
            user, song, label = line.strip().split('\t')
            track = song_track_dict[song]
            info_list1 = track_list_dict[track]
            popularity = popularity_dict[song]
            artist = song_artist_dict[song]
            is_user_like_artist = user_like_dict[user][artist] if artist in user_like_dict[user].keys() else 0.0
            year = song_year_dict[song]
            song_hottness = song_hottness_average if math.isnan(info_list1[-1]) else info_list1[-1]

            user_genre = user_like_genre_dict[user]
            genre_dict = song_genre_dict[song]
            user_like_genre_score = 0.0
            for k, v in genre_dict.items():
                if k in user_genre.keys():
                    user_like_genre_score += float(v) / 100 * float(user_genre[k])

            # X=[user,artist,is_user_like_artist,user_like_genre_score,year,popularity]+[duration,end_of_fade_in,loudness,mode,mode_confi,tempo,time_sig,time_sig_confi,artist_hottness,song_hottness]
            X.append([artist,float(is_user_like_artist),user_like_genre_score if user_like_genre_score!= 0.0 else 100.0,year_average if year == None else year,int(popularity)] + info_list1[:-1] + [song_hottness])
            Y.append(int(label))
            i += 1
            if i == 500000:
                break
    X = np.array(X)
    Y = np.array(Y)
    #print(X[:50])
    #convert to numerical
    df = pd.DataFrame(X)
    #df = df.apply(lambda x: pd.factorize(x)[0])
    stacked = df[[0]].stack()
    df[[0]] = pd.Series(stacked.factorize()[0],index=stacked.index).unstack()
    X = np.array(df,dtype=np.float64)[:50000]
    Y = Y[:50000]

    #one-hot
    enc = preprocessing.OneHotEncoder(n_values='auto',categorical_features=[0], dtype= np.float64)
    enc.fit(X)
    X = enc.transform(X).toarray()

    print(X.shape[1])
    return X,Y


def train_model_LR(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    X_train = list(X_train)
    y_train = list(y_train)

    # data augment
    # for i in range(1000):
    #     index = random.randint(0, len(X_train) - 1)
    #     sample = X_train[index]
    #     label = y_train[index]
    #     X_train.append(sample)
    #     y_train.append(label)

    # grid search
    params = {'penalty':['l2'],'C':[1.0,1.1,1.2,1.3,1.4,1.5]}
    model = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)
    model.fit(X_train,y_train)
    best_params = model.best_params_
    print('best_params',best_params)
    print('results',model.cv_results_)

    model = LogisticRegression(penalty='l2',solver='lbfgs',multi_class='ovr',C=1.5)
    model.fit(X_train,y_train)
    w = model.coef_[0]

    pred_y_prob = model.predict_proba(X_test)[:,1]
    pred_y = model.predict(X_test)

    #auc
    auc = roc_auc_score(y_test, pred_y_prob)
    print ('test-AUC: ', auc)
    # error
    score = model.score(X_test, y_test)
    print('error: ', 1-score)


if __name__ == '__main__':
    X,y = gen_x_y()
    #X,y = [[1,5],[0,5],[1,10],[1,4],[0,2],[0,4]],[1,0,1,1,0,0]
    train_model_LR(X, y)
