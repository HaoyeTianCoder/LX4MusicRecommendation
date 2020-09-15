import json
import pandas as pd
import numpy as np
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from config import *
import random
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import plot_tree
from graphviz import Digraph
import matplotlib.pyplot as plt

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
            for k, v in genre_dict.items():
                if k in user_genre.keys():
                    user_like_genre_score += float(v) / 100 * float(user_genre[k])

            # X=[user,artist,is_user_like_artist,user_like_genre_score,year,popularity]+[duration,end_of_fade_in,loudness,mode,mode_confi,tempo,time_sig,time_sig_confi,artist_hottness,song_hottness]
            X.append([artist,float(is_user_like_artist),user_like_genre_score if user_like_genre_score!= 0.0 else 100.0,year,int(popularity)]+info_list1)
            Y.append(int(label))
            i += 1
            if i == 500000:
                break

    ceate_feature_map(['artist','is_user_like_artist','user_like_genre_score','year','popularity']+['duration','end_of_fade_in','loudness','mode','mode_confi','tempo','time_sig','time_sig_confi','artist_hottness','song_hottness'])
    X = np.array(X)
    Y = np.array(Y)

    #print(X[:100])
    #print(X.shape)

    #convert to numerical
    df = pd.DataFrame(X)
    #df = df.apply(lambda x: pd.factorize(x)[0])
    stacked = df[[0]].stack()
    df[[0]] = pd.Series(stacked.factorize()[0],index=stacked.index).unstack()
    X = np.array(df,dtype=np.float64)
    #print(X[:100])

    return X,Y

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def train_model_xgb(X,Y,isAllFeat=False):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=40)

    # X_train = list(X_train)
    # y_train = list(y_train)

    # data augment
    for i in range(1000):
        index = random.randint(0, len(X_train) - 1)
        sample = X_train[index]
        label = y_train[index]
        #X_train.append(sample)
        #y_train.append(label)
        np.append(X_train,[sample],axis=0)
        np.append(y_train,[label],axis=0)

    if not isAllFeat:
        # extract continuous features
        X_train = X_train[:,1:]
        X_test = X_test[:,1:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 4, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic','gamma':1.3,'lambda':1.0,'subsample':1}
    param['eval_metric'] = 'error'
    watchlist = [ (dtrain, 'train'),(dtest, 'eval')]
    num_round = 200
    bst = xgb.train(param, dtrain, num_round, watchlist)
    bst.dump_model('xgb_model.raw.txt')
    #get weight of each leaf node
    with open('xgb_model.raw.txt','r',encoding='utf-8') as f:
        leaf_list = []
        for line in f:
            s = line.strip()
            # if s.find('booster') != -1:
            #     tree_index += 1
            if s.find('leaf') != -1:
                leaf_score = float(s.split('=')[1])
                leaf_list.append(leaf_score)
        #print('leaf_list: ',leaf_list)


    #show tree structure
    plot_tree(bst,fmap='xgb.fmap',num_trees=0)
    plt.show()

    #error
    # preds = bst.predict(dtest)
    # labels = dtest.get_label()
    # print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
    return bst

def splicing_feature(X,Y,bst,isAllFeat=False):

    if not isAllFeat:
        X_conti = X[:,1:]
        X_discrete = X[:,:1]

        dtrain_conti = xgb.DMatrix(X_conti, label=Y)
        leaf_id = bst.predict(dtrain_conti,pred_leaf=True)
        X = np.concatenate((leaf_id,X_discrete),axis=1)
        feature_num = X.shape[1]

        # one-hot
        enc = preprocessing.OneHotEncoder(n_values='auto', categorical_features=[i for i in range(feature_num-1)], dtype=np.float64)
        enc.fit(X)
        new_x = enc.transform(X).toarray()

        # add pre feature
        last = bst.predict(dtrain_conti)
        last = last.reshape(len(X), 1)
        new_x = np.concatenate((new_x, last), axis=1)
        print('after one-hot, dimension:', new_x.shape[1])

        # drop low cover feature point
        drop_index = []
        for column in range(len(new_x[0])):
            zero_num = 0
            for row in range(len(new_x)):
                if new_x[row][column] == 0:
                    zero_num += 1
            if zero_num >= 0.9 * len(new_x):
                drop_index.append(column)
        print('drop length: ', len(drop_index))
        for column in drop_index:
            new_x[:, column] = np.zeros(len(new_x))
        print('after drop: ', new_x.shape[1] - len(drop_index))
    else:
        dtrain = xgb.DMatrix(X, label=Y)
        leaf_id = bst.predict(dtrain,pred_leaf=True)
        # one-hot
        enc = preprocessing.OneHotEncoder(n_values='auto',dtype=np.float64)
        enc.fit(leaf_id)
        new_x = enc.transform(leaf_id).toarray()

        #add pre feature
        last = bst.predict(dtrain)
        last = last.reshape(len(X),1)
        new_x = np.concatenate((new_x,last),axis=1)
        print('after one-hot, dimension:',new_x.shape[1])

        #drop low cover feature point
        drop_index = []
        for column in range(len(new_x[0])):
            zero_num = 0
            for row in range(len(new_x)):
                if new_x[row][column] == 0:
                    zero_num += 1
            if zero_num >= 0.9*len(new_x):
                drop_index.append(column)
        print('drop length: ', len(drop_index))
        for column in drop_index:
            new_x[:,column] = np.zeros(len(new_x))
        print('after drop: ', new_x.shape[1] - len(drop_index))

    return new_x,Y

def train_model_LR(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

    X_train = list(X_train)
    y_train = list(y_train)

    # data augment
    # for i in range(3000):
    #     index = random.randint(0, len(X_train) - 1)
    #     sample = X_train[index]
    #     label = y_train[index]
    #     X_train.append(sample)
    #     y_train.append(label)

    #grid search
    # params = {'penalty':['l1','l2'],'C':[1.0,1.1,1.2,1.3,1.4,1.5]}
    # model = GridSearchCV(LogisticRegression(), param_grid=params, cv=5)
    # model.fit(X_train,y_train)
    # best_params = model.best_params_
    # print('best_params',best_params)

    model = LogisticRegression(penalty='l2',solver='lbfgs',multi_class='ovr',C=1.5)
    model.fit(X_train,y_train)
    w = model.coef_[0]
    #print('w:', w)

    print('xgboost + lr -->')
    pred_y = model.predict_proba(X_test)[:,1]

    #auc
    auc = roc_auc_score(y_test, pred_y)
    print ('AUC: ', auc)
    # error
    score = model.score(X_test, y_test)
    print('error: ', 1-score)

    return model


if __name__ == '__main__':
    isAllFeat = True

    X,Y = gen_x_y()
    bst = train_model_xgb(X,Y,isAllFeat)
    new_x,new_y = splicing_feature(X,Y,bst,isAllFeat)
    lr = train_model_LR(new_x,new_y)
