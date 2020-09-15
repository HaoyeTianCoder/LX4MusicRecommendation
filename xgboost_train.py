import json
import pandas as pd
import numpy as np
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from config import *
import random
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt


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

            user_genre = user_like_genre_dict[user]
            genre_dict = song_genre_dict[song]
            user_like_genre_score = 0.0
            for k,v in genre_dict.items():
                if k in user_genre.keys():
                    user_like_genre_score += float(v)/100 * float(user_genre[k])

            # X=[user,artist,is_user_like_artist,user_like_genre_score,year,popularity]+[duration,end_of_fade_in,loudness,mode,mode_confi,tempo,time_sig,time_sig_confi,artist_hottness,song_hottness]
            X.append([artist, float(is_user_like_artist),user_like_genre_score if user_like_genre_score!= 0.0 else 100.0, year, int(popularity)] + info_list1)
            Y.append(int(label))
            i += 1
            if i == 500000:
                break
    X = np.array(X)
    Y = np.array(Y)

    #print(X[:50])
    #print(X.shape)

    #convert to numerical
    df = pd.DataFrame(X)
    #df = df.apply(lambda x: pd.factorize(x)[0])
    stacked = df[[0]].stack()
    df[[0]] = pd.Series(stacked.factorize()[0],index=stacked.index).unstack()
    X = np.array(df,dtype=np.float64)
    #print(X[:100])


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=45)

    X_train = list(X_train)
    y_train =list(y_train)

    #data augment
    for i in range(1000):
        index = random.randint(0,len(X_train)-1)
        sample = X_train[index]
        label = y_train[index]
        X_train.append(sample)
        y_train.append(label)

    return X_train, X_test, y_train, y_test


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def train_model_xgb(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 5, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic','gamma':1.0,'lambda':1.1,'subsample':1}
    param['eval_metric'] = 'auc'
    watchlist = [ (dtrain, 'train'),(dtest, 'eval')]
    num_round = 300
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=5)

    preds = bst.predict(dtest)
    labels = dtest.get_label()
    print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

def train_model_xgb_cv(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_sklearn = XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=3,
        min_child_weight=1,
        gamma=0.3,
        subsample=0.6,
        colsample_bytree=0.7,
        objective='binary:logistic',
        nthread=4,
        seed=27,
        reg_lambda=0.01
    )

    xgb_params = xgb_sklearn.get_params()
    cvresult = xgb.cv(xgb_params, dtrain, num_boost_round=xgb_params['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=5)
    n_estimators = cvresult.shape[0]
    print("n_estimators: ",n_estimators)
    xgb_sklearn.set_params(n_estimators=n_estimators)
    xgb_sklearn.fit(np.array(X_train),np.array(y_train),eval_metric='auc')

    pred_y = xgb_sklearn.predict(X_test)
    pred_y_prob = xgb_sklearn.predict_proba(X_test)[:,1]
    # auc
    auc = roc_auc_score(y_test, pred_y_prob)
    print('AUC: ', auc)
    # error
    score = xgb_sklearn.score(X_test, y_test)
    print('error: ',1-score)



    # grid search
    params = {'max_depth':[2,3,4,5,6,7,8]}
    model = GridSearchCV(estimator=XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        # max_depth=3,
        min_child_weight=1,
        gamma=0.3,
        subsample=0.6,
        colsample_bytree=0.7,
        objective='binary:logistic',
        nthread=4,
        seed=27,
        reg_lambda=0.01
    ),param_grid=params,cv=2)
    model.fit(np.array(X_train),np.array(y_train),eval_metric='auc')
    print(model.cv_results_,model.best_params_,model.best_score_)


    feat_imp = pd.Series(xgb_sklearn.get_booster().get_fscore(fmap='xgb.fmap')).sort_values(ascending=True)
    feat_imp.plot(kind='barh',color='black',legend=False, figsize=(10, 6))
    plt.ylabel('Feature name')
    plt.xlabel('Feature score')
    plt.savefig('C:/Users/Administrator.USER-20161227PQ/Desktop/paper figure/figure5.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    ceate_feature_map(['artist','prefer_score_artist','prefer_score_genre','release_year','popularity']+['duration','end_of_fade_in','loudness','song_mode','mode_confi','tempo','time_sig','time_sig_confi','artist_hottness','song_hottness'])
    X_train, X_test, y_train, y_test = gen_x_y()
    #train_model_xgb(X_train, X_test, y_train, y_test)
    train_model_xgb_cv(X_train, X_test, y_train, y_test)