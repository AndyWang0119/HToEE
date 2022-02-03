#import stuff
import argparse
import pandas as pd
import glob
import numpy as np
import scipy as sp
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance
import xgboost as xgb
import pickle
import yaml
from keras import Sequential
from keras.layers import Dense, Dropout
import keras     
from scipy import integrate
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import math

#loading data
parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('Required Arguments')
required_args.add_argument('-b','--bdt', action='store', help='Input bdt dataframe dir', required=True)
options=parser.parse_args()

dataframe = pd.read_csv('%s/data.csv'%options.bdt)
print('successfully loaded data')


def sb_equalise_weight(df):
    '''
    equalise the signal background weights of the dataframe
    '''
    sig_weight = df.loc[df['y']==1, 'weights']
    bkg_weight = df.loc[df['y']==0, 'weights']
    print(np.sum(sig_weight), np.sum(bkg_weight))
    #b_to_s_ratio = np.sum(bkg_weight)/np.sum(sig_weight)
    df.loc[df['y']==1, 'weights'] = sig_weight / sum(sig_weight)
    df.loc[df['y']==0, 'weights'] = bkg_weight / sum(bkg_weight)
    return df

dataframe = sb_equalise_weight(dataframe)
#dataframe = sb_equalise_weight(dataframe)


Gev_list = ['diphotonPt', 'dijetMass', 'leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt']
def take_log(df, list):
    '''
    a function that take the log of GeV unit value to scale them down
    '''
    for feature in list:
        df.loc[df[feature]!=-999.0, feature] = np.log(df.loc[df[feature]!=-999.0, feature])
    return df

dataframe = take_log(dataframe, Gev_list)

#define y train test set and weights
y_train = dataframe[dataframe['type']=='train']['y']
y_pred_train = dataframe[dataframe['type']=='train']['y_pred']
train_weights = dataframe[dataframe['type']=='train']['weights']
#train_weights_eq = dataframe[dataframe['type']=='train']['weights_eq']

y_test = dataframe[dataframe['type']=='test']['y']
y_pred_test = dataframe[dataframe['type']=='test']['y_pred']
test_weights = dataframe[dataframe['type']=='test']['weights']
#test_weights_eq = dataframe[dataframe['type']=='test']['weights_eq']

#sb_eq = 1
#if sb_eq:
#    train_weights = train_weights_eq
#    test_weights = test_weights_eq

features =     ['diphotonPt','diphotonCosPhi' , 
    'leadPhotonPtOvM', 'subleadPhotonPtOvM',
    'leadPhotonEta', 'subleadPhotonEta',
    'dijetAbsDEta', 'dijetDPhi','dijetMinDRJetPho', 'dijetMass', 
    'dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'dijetCentrality',
    'leadJetDiphoDPhi', 'subleadJetDiphoDPhi', 'leadJetDiphoDEta', 'subleadJetDiphoDEta',
    'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', 
    'subleadJetEn', 'subleadJetPt', 'subleadJetEta', 'subleadJetPhi','subleadJetQGL',
    'leadPhotonIDMVA', 'subleadPhotonIDMVA',
    ]


#define x train tests
x_train = dataframe[dataframe['type']=='train'][features]
x_test = dataframe[dataframe['type']=='test'][features]

import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
plt.style.use("cms10_6_HP")
import matplotlib as mpl

def plot_roc(y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = False, label = '', color='red'):

    if abs:
        train_weights = np.abs(train_weights)
        test_weights = np.abs(test_weights)

    bkg_eff_train, sig_eff_train, _ = roc_curve(y_train, y_pred_train, sample_weight=train_weights)
    bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test, sample_weight=test_weights)

    fig = plt.figure(1)
    axes = fig.gca()
    axes.plot(bkg_eff_train, sig_eff_train, color=color, label='Train %s'%label)
    axes.plot(bkg_eff_test, sig_eff_test, color=color, label='Test %s'%label, ls='--')
    axes.set_xlabel('Background efficiency', ha='right', x=1, size=13)
    axes.set_xlim((0,1))
    axes.set_ylabel('Signal efficiency', ha='right', y=1, size=13)
    axes.set_ylim((0,1))
    axes.legend(ncol=1, prop={'size':10}, loc='upper right')
    axes.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)


def calculate_auc(bkg_eff, sig_eff):
    auc = integrate.trapz(x=bkg_eff, y=sig_eff)
    return auc 

roc = 0
if roc:
    plot_roc(y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = False)
    plt.savefig('plotting/plots/ggH_BDT/ROC_') 
    plt.close()
    plot_roc(y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = True, label='absolute', color='blue') 
    plt.savefig('plotting/plots/ggH_BDT/ROC_absolute') 
    plt.close()

important = 0
if important:
    clf = pickle.load(open('models/ggH_BDT_clf.pickle.dat'))

    clf.get_booster().feature_names = features

    for type in ['gain', 'weight', 'cover']:
        ax = plot_importance(clf, importance_type=type, show_values=True, title='Importance by %s'%type) 

        fig = ax.figure
        plt.savefig('plotting/plots/ggH_BDT/feature_importance_%s.png'%type)

ks_test = 0
if ks_test:
    #get xrange from yaml config
    with open('plotting/var_to_xrange.yaml', 'r') as plot_config_file:
        plot_config        = yaml.load(plot_config_file)
        var_to_xrange = plot_config['var_to_xrange']

    columns = dataframe.columns
    ks_df = []
    for col in columns:
        if col != 'type' and col != 'weights':
            print(col)

            train_set = dataframe[dataframe['type']=='train'][col]
            train_set_weight = dataframe[dataframe['type']=='train']['weights']
            test_set = dataframe[dataframe['type']=='test'][col]
            test_set_weights = dataframe[dataframe['type']=='test']['weights']

            D, p = sp.stats.ks_2samp(train_set, test_set)
            ks_df.append(p)
            print(D,p)

            fig = plt.figure(1)
            axes = fig.gca()

            try: bins = np.linspace(var_to_xrange[col][0], var_to_xrange[col][1], 40)
            except KeyError: bins = np.linspace(min(train_set), max(train_set), 40)

            #axes.hist(train_set, bins=bins, normed=True, cumulative=True, label='CDF Train', histtype='step')
            #axes.hist(test_set, bins=bins, normed=True, cumulative=True, label='CDF Test', histtype='step')

            #ratio plot
            train_binned, train_bin_edge = np.histogram(train_set, bins=bins, weights=train_set_weight/sum(train_set_weight))
            test_binned, test_bin_edge = np.histogram(test_set, bins=bins, weights=test_set_weights/sum(test_set_weights))

            bin_centres = (train_bin_edge[:-1] + train_bin_edge[1:])/2
            axes.plot(bin_centres, train_binned/test_binned, label='tran/test ratio')
            
            col = col.replace('_',' ')
            #axes.set_ylabel('CDF Prob')
            axes.set_ylabel('Ratio')
            axes.set_xlabel(col)
            axes.set_title('p value: %s'%p)
            axes.set_ylim(0, 2)
            axes.legend(bbox_to_anchor=(0.97,0.97))
            plt.savefig('plotting/plots/ggH_BDT/ks/ks_%s_%s.png' %(col,round(p,3)))
            plt.close()
    ks_df.append(None)
    ks_df.append(None)
    ks_df = pd.DataFrame(ks_df, index=columns)
    ks_df.to_csv('plotting/plots/ggH_BDT/ks/ks.csv')

#perform a 1D search with plots
def find_eff(eff, value = 0.9):
    '''
    find the first index of eff whose value is larger than the given
    '''
    index = -1
    for i in eff:
        index += 1
        if i > value:
            return index

import random
def cross_validate(df, fold=3):
    index = np.arange(0,len(df))
    random.shuffle(index)
    print(index)
    fold_index_list = np.array_split(index, fold)
    data_fold = []
    for f in fold_index_list:
        data_fold.append(df.iloc[f.tolist()])
    
    return data_fold



one_cv = 0
if one_cv:
    grid_rnge     = {'learning_rate': [0.01, 0.05, 0.1, 0.3,1],
                    'max_depth':[1,2,3,4,5,10,20],
                    'min_child_weight':[x for x in range(0,10)],
                    'gamma': np.arange(0,5.5,0.5).tolist(),
                    'subsample': [0,0.1,0.2,0.3,0.4,0.5, 0.8, 1.0],
                    #'n_estimators':[5,10]}
                    'n_estimators':[5,10,20,30,50,100,150,300,500]}
    
    grid_rnge_default     = {'learning_rate': 0.1,
                    'max_depth':4,
                    'min_child_weight':0,
                    'gamma': 1,
                    'subsample': 0.6,
                    'n_estimators':100}

    fold = 3
    data_fold = cross_validate(dataframe, fold=fold)
    
    #for hp in ['max_depth','min_child_weight','subsample','gamma','n_estimators']:
    for hp in ['min_child_weight']:
        score_list = []
        print("1D search for %s" %hp)
        dic = grid_rnge_default.copy()

        bkg_eps_list = []
        for value in grid_rnge[hp]:
            print('%s is set to %s' %(hp,value))
            dic[hp] = value

            #k-fold validation
            bkg_eps_each_fold = []
            for i in range(fold):
                print('running %sth fold' %i)
                x_test =  data_fold[i][features]
                y_test = data_fold[i]['y']
                test_weights = data_fold[i]['weights']

                a = np.arange(0,fold).tolist()
                a.remove(i)
                train = pd.concat(data_fold[j] for j in a)
                x_train = train[features]
                y_train = train['y']
        
                #load already trained models
                try: 
                    clf = pickle.load(open('Andy/models/{}.pickle.dat'.format(hp+'_'+str(value)+'_'+str(i)+'of'+str(fold))))
                    print('sucessufuly loaded model')
                except IOError:
                    clf = xgb.XGBClassifier(objective='binary:logistic', 
                            n_estimators=dic['n_estimators'], 
                            learning_rate=dic['learning_rate'], 
                            max_depth=dic['max_depth'], 
                            min_child_weight=dic['min_child_weight'], 
                            subsample=dic['subsample'], 
                            gamma=dic['gamma'])
                    clf.fit(x_train,y_train)
                    pickle.dump(clf, open("Andy/models/{}.pickle.dat".format(hp+'_'+str(value)+'_'+str(i)+'of'+str(fold)), "wb"))

                y_pred_test = clf.predict_proba(x_test)[:,1:]
                bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test, sample_weight=test_weights)
                
                bkg_eps = bkg_eff_test[find_eff(sig_eff_test, 0.9)]
                bkg_eps_each_fold.append(bkg_eps)

            bkg_eps_list.append(np.average(bkg_eps_each_fold))
        
        fig = plt.figure(1)
        axe = plt.gca()
        axe.scatter(grid_rnge[hp], bkg_eps_list)
        axe.set_xlabel(hp.replace('_',' '))
        axe.set_ylabel('background efficiency')
        diff = max(bkg_eps_list) - min(bkg_eps_list)
        axe.set_ylim(min(bkg_eps_list) - diff, max(bkg_eps_list) + diff)
        axe.set_title('1D HP Search')
        plt.savefig('plotting/plots/ggH_BDT/one_dim_HP_%s.png'%hp)
        plt.close()
        

many_cv = 0
if many_cv:
    clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, 
                                                eta=0.05, maxDepth=4, min_child_weight=0.01, 
                                                subsample=0.6, colsample_bytree=0.6, gamma=1)
    grid_rnge     = {#'learning_rate': [0.01, 0.05, 0.1, 0.3],
                        'max_depth':[x for x in range(3,10)],
                        'min_child_weight':[x for x in range(0,3)],
                        #'gamma': np.linspace(0,5,6).tolist(),
                        #'subsample': [0.5, 0.8, 1.0],
                        #'n_estimators':[200,300,400,500]
                    }
    grid = GridSearchCV(clf, grid_rnge, cv=3, n_jobs=-1, verbose=2)
    grid.fit(x_train, y_train)
    print('best parameters:', grid.best_params_)


plot_output = 1
if plot_output:

    #x_train = x_train.replace(-999.0,-10)
    #x_test = x_test.replace(-999.0,-10)
    
    scaler = StandardScaler()
    #x_train = scaler.fit_transform(x_train)
    #x_test = scaler.transform(x_test)  


    fig  = plt.figure(1)
    axes = fig.gca()       
    bins = np.linspace(0,1,41)

    
    sig_w_true = test_weights.ravel() * (y_test==1)
    bkg_w_true = test_weights.ravel() * (y_test==0)
    
    '''
    sig_w_true_train = train_weights.ravel() * (y_train==1)
    bkg_w_true_train = train_weights.ravel() * (y_train==0)
    bkg_w_true_train /= np.sum(bkg_w_true_train)
    sig_w_true_train /= np.sum(sig_w_true_train)

    '''


    normalise = True
    if normalise:
        sig_w_true /= np.sum(sig_w_true)
        bkg_w_true /= np.sum(bkg_w_true)

        
    try: 
        clf = pickle.load(open('Andy/models/best_model_preprocess.pickle.dat'))
        print('sucessufuly loaded model')
    except IOError:
        print('training a BDT')
        clf = xgb.XGBClassifier(objective='binary:logistic', 
                n_estimators=200, 
                learning_rate=0.25, 
                max_depth=10, 
                subsample=0.2) 
        clf.fit(x_train,y_train,train_weights)
        pickle.dump(clf, open("Andy/models/best_model_preprocess.pickle.dat",  "wb"))

    try: 
        clf2 = pickle.load(open('Andy/models/default_model.pickle.dat'))
        print('sucessufuly loaded model')
    except IOError:
        print('training a BDT')
        clf2 = xgb.XGBClassifier(objective='binary:logistic')
        clf2.fit(x_train,y_train,train_weights)
        pickle.dump(clf2, open("Andy/models/default_model.pickle.dat",  "wb"))

    y_pred_test_op = clf.predict_proba(x_test)[:,1:]
    y_pred_train_op = clf.predict_proba(x_train)[:,1:]
    sig_scores_op = y_pred_test_op.ravel()  * (y_test==1)
    bkg_scores_op = y_pred_test_op.ravel()  * (y_test==0)

    #y_pred_test = clf2.predict_proba(x_test)[:,1:]
    #y_pred_train = clf2.predict_proba(x_train)[:,1:]
    #sig_scores = y_pred_test.ravel()  * (y_test==1)
    #bkg_scores = y_pred_test.ravel()  * (y_test==0)

    #axes.hist(sig_scores, bins=bins, label='ggH default', weights=sig_w_true, histtype='step', color = 'blue')
    #axes.hist(bkg_scores, bins=bins, label='VBF default', weights=bkg_w_true, histtype='step', stacked=True, zorder=0, color = 'orange')

    axes.hist(sig_scores_op, bins=bins, label='ggH HP', weights=sig_w_true, histtype='step', color = 'blue', ls='dotted')
    axes.hist(bkg_scores_op, bins=bins, label='VBF HP', weights=bkg_w_true, histtype='step', stacked=True, zorder=0, color = 'orange',ls='dotted')

  
    axes.legend(bbox_to_anchor=(0.9,0.97), ncol=2, prop={'size':10})
    if normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
    else: axes.set_ylabel('Events', ha='right', y=1, size=13)


    axes.set_xlabel('Score', ha='right', x=1, size=13)


    current_bottom, current_top = axes.get_ylim()

    axes.set_yscale('log', nonposy='clip')

    axes.set_ylim(top=current_top*2)

    plt.savefig('plotting/plots/ggH_BDT/output_score_RDF.png')

    plt.close()
    
    #plot_roc(y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = False, label='default')
    plot_roc(y_train, y_pred_train_op, train_weights, y_test, y_pred_test_op, test_weights, abs = False, label='with HP', color='blue')
    plt.savefig('plotting/plots/ggH_BDT/roc_compare_hp.png')

    bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test_op, sample_weight=test_weights)
    bkg_eps = bkg_eff_test[find_eff(sig_eff_test, 0.9)]
    print('bkg efficiency with signal eff 0.9: %s' %bkg_eps)
    print('auc: %s' %calculate_auc(bkg_eff_test, sig_eff_test))


plot_output_ann = 0
if plot_output_ann:
    x_train_cp = x_train.copy()
    x_test_cp = x_test.copy()
    x_train = x_train.replace(-999.0,-10)
    x_test = x_test.replace(-999.0,-10)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  
    scaler2 = StandardScaler()
    x_train_cp = scaler2.fit_transform(x_train_cp)
    x_test_cp = scaler2.transform(x_test_cp)  

    fig  = plt.figure(1)
    axes = fig.gca()       
    bins = np.linspace(0,1,41)

    
    sig_w_true = test_weights.ravel() * (y_test==1)
    bkg_w_true = test_weights.ravel() * (y_test==0)

    normalise = True

    epoch = 100
    hidden = 2 #number of hidden layers
    dropout = 0 #if add dropout before the first layer
    neurons = 10 #number of neurons each layer
    batch_size = 64
    regulation = 0.001
    start_lr = 0.001
    patience = 10
    exp_decay = 0.05
    decay = 0.1
    ann_name = 'ann_definite'
    compare_name = None
    compare_name2 = None

    #scaler for weights
    weight_scaler = 10**6

    have_done = ['ann_10', 'ann_10_10', 'ann_dropout_10_10', 'ann_10_10_10', 'ann_10_10_10_10',
                'ann_15_30eq', 'ann_15_10e_withweights',
                'ann_10_10_30e', 'ann_dropout_10_10_30e', 'ann_10_10_10_30e', 'ann_10_10_10_10_30e', 
                'ann_100_30e', 'ann_dropout_10_30e', 'ann_100',
                'ann_100_100_100e', 'ann_100_100_100e_64b', 'ann_100_100_100e_64b_adam', #scale the weight from now
                'ann_200_200_100e', 'ann_dropout_100_100_100e', 'ann_100_100_100e_reg_Nadam',
                'ann_100_100_100e_reg_sb', 'ann_100_100_20e_reg_sb', 'ann_100_100_100e_reg_sb_noreg','ann_100_100_20e_reg_sb_hireg', #after sb equl 
                'ann_100_100_20e_reg_sb_nosgd', 'ann_10_10_100e_reg_sb_nosgd',
                'ann_10_10_es_sb', 'ann_10_10_es_sb_lr01', 'ann_10_10_es_sb_lr0005_reg005', 'ann_10_10_es_sb_lr0005_pat10',
                'ann_10_10_es_sb_lr0005_noreg_pat10', 'ann_10_10_es_sb_lr0005_noreg',
                'ann_10_10_l1',
                'ann_10_10_sch_200e_defi_noreg', 'ann_10_10_sch_200e_defi', 'ann_10_10_sch_200e_defi_l2',
                'ann_10_10_sch_200e_defi_nsb',
                'ann_10_10_sch_200e_defi_minus'] 

    print('doing for %s' %ann_name)

    # Define configuration parameters
    
    # Define the scheduling function
    def schedule(epoch):
        def lr(epoch, start_lr, exp_decay):
            return start_lr * math.exp(-exp_decay*epoch)
        if epoch < 5:
            return start_lr
        else:
            return lr(epoch-4, start_lr, exp_decay)
    def schedule2(epoch):
        def lr(epoch, start_lr, decay):
            #nonlocal previous_lr
            return start_lr * 1/(1+exp_decay*epoch)
        return lr(epoch, start_lr, decay)

    if normalise:
        sig_w_true /= np.sum(sig_w_true)
        bkg_w_true /= np.sum(bkg_w_true)

    try: 
        model = pickle.load(open('Andy/models/%s.pickle.dat' %ann_name))
        print('sucessufuly loaded ANN model')
        train_model = 0
    except IOError:
        train_model = 1
        model = Sequential()
        model.add(Dense(len(features), activation='relu', input_dim = len(features)))

        for i in range(hidden):
            print('add 1 hidden layer')
            model.add(Dense(neurons, activation='relu',activity_regularizer=keras.regularizers.l1(regulation)))
            #model.add(Dense(neurons, activation='relu'))
            if dropout:
                model.add(Dropout(0.2))
 
        model.add(Dense(1, activation='sigmoid'))
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
        #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
        #optimizer = keras.optimizers.Adam(lr = 0.001)
        optimizer = keras.optimizers.Nadam()
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        print(model.summary())
        #x_train, x_val, y_train, y_val, train_weights, val_weights = train_test_split(x_train, y_train, train_weights, test_size=0.15, shuffle=True)
        print('training a BDT')
        hist = model.fit(x_train,y_train.to_numpy(),sample_weight=train_weights*weight_scaler, batch_size=batch_size, epochs=epoch, verbose=1,
        validation_data=(x_test, y_test.to_numpy(),test_weights.to_numpy()*weight_scaler),
        #validation_split=0.1,
        callbacks = [keras.callbacks.LearningRateScheduler(schedule, verbose=1), early_stopping],
        )
        pickle.dump(model, open("Andy/models/%s.pickle.dat" %ann_name,  "wb"))
    
    #load a comparison model
    if compare_name != None:
        model_comp = pickle.load(open('Andy/models/%s.pickle.dat' %compare_name))
 

    #get prediction for sig and background
    y_pred_test_nn = model.predict_proba(x_test)
    sig_scores_nn = y_pred_test_nn.ravel()  * (y_test==1)
    bkg_scores_nn = y_pred_test_nn.ravel()  * (y_test==0)

    y_pred_train_nn = model.predict_proba(x_train)

    if compare_name != None:
        y_pred_test_comp = model_comp.predict_proba(x_test)
        y_pred_test_comp = model_comp.predict_proba(x_test_cp)
        sig_scores_comp = y_pred_test_comp.ravel()  * (y_test==1)
        bkg_scores_comp = y_pred_test_comp.ravel()  * (y_test==0)
        y_pred_train_comp = model_comp.predict_proba(x_train)
        y_pred_train_comp = model_comp.predict_proba(x_train_cp)

    #plot the output score
    axes.hist(sig_scores_nn, bins=bins, label='ggH %s'%ann_name.replace('_', ' '), weights=sig_w_true, histtype='step', color = 'blue')
    axes.hist(bkg_scores_nn, bins=bins, label='VBF %s'%ann_name.replace('_', ' '), weights=bkg_w_true, histtype='step', stacked=True, zorder=0, color = 'orange')
    
    if compare_name != None:
        axes.hist(sig_scores_comp, bins=bins, label='ggH %s'%compare_name.replace('_', ' '), weights=sig_w_true, histtype='step', color = 'blue', ls='dotted')
        axes.hist(bkg_scores_comp, bins=bins, label='VBF %s'%compare_name.replace('_', ' '), weights=bkg_w_true, histtype='step', stacked=True, zorder=0, color = 'orange', ls='dotted')
           
    axes.legend(ncol=2, prop={'size':10}, loc='upper left')
    if normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
    else: axes.set_ylabel('Events', ha='right', y=1, size=13)
    axes.set_xlabel('Score', ha='right', x=1, size=13)
    current_bottom, current_top = axes.get_ylim()
    axes.set_yscale('log', nonposy='clip')
    axes.set_ylim(top=current_top*2)
    
    if compare_name != None:
        plt.savefig('plotting/plots/ggH_BDT/output_score_ANN_%s_against_%s.png' %(ann_name,compare_name))
    else:
        plt.savefig('plotting/plots/ggH_BDT/output_score_ANN_%s.png' %ann_name)
    plt.close()

    #plot the roc curve
    plot_roc(y_train, y_pred_train_nn, train_weights, y_test, y_pred_test_nn, test_weights, abs = False, label=ann_name.replace('_', ' '), color='blue')
    
    if compare_name != None:
        plot_roc(y_train, y_pred_train_comp, train_weights, y_test, y_pred_test_comp, test_weights, abs = False, label=compare_name.replace('_', ' '), color='red')
        plt.savefig('plotting/plots/ggH_BDT/roc_ann_%s_against_%s.png'%(ann_name,compare_name))
    else:
        plt.savefig('plotting/plots/ggH_BDT/roc_ann_%s.png'%ann_name)
    plt.close()

    bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test_nn, sample_weight=test_weights*weight_scaler)
    bkg_eps = bkg_eff_test[find_eff(sig_eff_test, 0.9)]
    print('bkg efficiency with signal eff 0.9: %s' %round(bkg_eps,4))
    print('auc: %s' %round(calculate_auc(bkg_eff_test, sig_eff_test),4))

    #plot training history
    if train_model == 0:
        history_ann = pd.read_csv('Andy/train_history/%s.csv' %ann_name)
    else:
        history_ann = hist.history
        #save history
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv('Andy/train_history/%s.csv' %ann_name)

    plt.plot(history_ann['loss'], label='loss %s'%ann_name.replace('_',' '), color='blue')
    plt.plot(history_ann['val_loss'], label='val loss %s'%ann_name.replace('_',' '), color='orange')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    if compare_name != None:
        history_comp = pd.read_csv('Andy/train_history/%s.csv' %compare_name)
        plt.plot(history_comp['loss'], label='loss %s'%compare_name.replace('_',' '), color='blue', ls='--')
        plt.plot(history_comp['val_loss'], label='val loss %s'%compare_name.replace('_',' '), color='orange', ls='--')
        plt.legend(loc='upper right') 
        plt.savefig('plotting/plots/ggH_BDT/training_history_%s_against_%s' %(ann_name,compare_name))
    else:
        plt.legend(loc='upper right') 
        plt.savefig('plotting/plots/ggH_BDT/training_history_%s' %ann_name)

    plt.close()


dnn = 0
if dnn:


    from sklearn.preprocessing import StandardScaler
    print('Trianing a ANN')
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)                                      


    model = Sequential()
    model.add(Dense(len(features), activation='relu', input_dim = len(features)))
    #model.add(Dense(10, activation='relu'))
    #model.add(Dense(10, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])


    print(model.summary())

    #model.fit(x_train, y_train.to_numpy(), batch_size=10, epochs=10, verbose=1)
    model.fit(x_train,y_train.to_numpy(),sample_weight=train_weights, batch_size=10, epochs=30, verbose=1)

    y_pred_test = model.predict_proba(x_test)

    bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test, sample_weight=test_weights)
    bkg_eps = bkg_eff_test[find_eff(sig_eff_test, 0.9)]
    print('bkg efficiency with signal eff 0.9: %s' %round(bkg_eps,4))

    plot_roc(y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = False, label='default')

rnn = 0
if rnn:
    n_lstm_layers=3
    n_lstm_nodes=150
    n_dense_1=1
    n_nodes_dense_1=300
    n_dense_2=4
    n_nodes_dense_2=200
    dropout_rate=0.1
    learning_rate=0.001
    batch_norm=True


    input_objects = keras.layers.Input(shape=(2, 8), name='input_objects') 
    input_global  = keras.layers.Input(shape=(13,), name='input_global')
    lstm = input_objects
    decay = 0.2
    for i_layer in range(n_lstm_layers):
        #lstm = keras.layers.LSTM(n_lstm_nodes, activation='tanh', kernel_regularizer=keras.regularizers.l2(decay), recurrent_regularizer=keras.regularizers.l2(decay), bias_regularizer=keras.regularizers.l2(decay), return_sequences=(i_layer!=(n_lstm_layers-1)), name='lstm_{}'.format(i_layer))(lstm)
        lstm = keras.layers.LSTM(n_lstm_nodes, activation='tanh', return_sequences=(i_layer!=(n_lstm_layers-1)), name='lstm_{}'.format(i_layer))(lstm)

    #inputs to dense layers are output of lstm and global-event variables. Also batch norm the FC layers
    dense = keras.layers.concatenate([input_global, lstm])
    for i in range(n_dense_1):
        dense = keras.layers.Dense(n_nodes_dense_1, activation='relu', kernel_initializer='he_uniform', name = 'dense1_%d' % i)(dense)
        if batch_norm:
            dense = keras.layers.BatchNormalization(name = 'dense_batch_norm1_%d' % i)(dense)
    dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout1_%d' % i)(dense)

    for i in range(n_dense_2):
        dense = keras.layers.Dense(n_nodes_dense_2, activation='relu', kernel_initializer='he_uniform', name = 'dense2_%d' % i)(dense)
        #add droput and norm if not on last layer
        if batch_norm and i < (n_dense_2 - 1):
            dense = keras.layers.BatchNormalization(name = 'dense_batch_norm2_%d' % i)(dense) 
        if i < (n_dense_2 - 1):
            dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout2_%d' % i)(dense)

    output = keras.layers.Dense(1, activation = 'sigmoid', name = 'output')(dense)
    #optimiser = keras.optimizers.Nadam(lr = learning_rate)
    optimiser = keras.optimizers.Adam(lr = learning_rate)

    model = keras.models.Model(inputs = [input_global, input_objects], outputs = [output])
    model.compile(optimizer = optimiser, loss = 'binary_crossentropy')
    print(model.summary())