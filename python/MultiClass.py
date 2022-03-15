import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
import xgboost as xgb
import pickle
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
import keras     
from scipy import integrate
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import math
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
plt.style.use("cms10_6_HP")
#import matplotlib as mpl
import seaborn as sn
import gc
#from memory_profiler import profile
#%load_ext 


class multiClass(object):
    def __init__(self, dataframe_full, train_vars, 
    folder, class_list, stxs_list, color_map, data_label='', recon=0):
        '''
        :param class_list: class labels that want the classifier to classify
        :type stxs_list: list
        :param stxs_list: actual HTXS_1.2 label from the ROOT file
        :type stxs_list: list
        '''
        self.dataframe = dataframe_full
        self.folder = folder
        self.class_list = class_list
        self.stxs_list = stxs_list
        self.color_map = color_map

        self.train_vars = train_vars
        self.train_vars_scale = list(self.train_vars)

        self.classifer = None
        self.classifer_comp = None

        self.equalised = False

        self.data_label = data_label


        self.loadcomplieddata = recon


    def combine_dataframe(self, prod=False):
        '''
        assign signal type to the dataframe;
        combine them into a single dataframe;
        '''
        if self.loadcomplieddata:
            print(1)
            self.dataframe['signal'] = self.dataframe['proc']
            print(1)
            for i, signal in enumerate(self.stxs_list):
                i += 200
                self.dataframe.loc[self.dataframe.HTXS_stage1_2_cat_pTjet30GeV==i, 'signal'] = signal
            print(1)
            self.dataframe['bkg'] = np.ones(len(self.dataframe))
   
            print(1)
            #remove not wanted stxs from vbf vh
            for signal in ['QQ2HQQ_FWDH', 'VH', 'VBF']:
                #self.dataframe = self.dataframe.loc[self.dataframe.signal != signal]
                self.dataframe = self.dataframe[self.dataframe['signal']!=signal]
            print(1)

            #conbine the 0 1 jet stxs bins
            for signal in ['QQ2HQQ_0J', 'QQ2HQQ_1J', 'QQ2HQQ_GE2J_MJJ_0_60', 'QQ2HQQ_GE2J_MJJ_120_350']:
                self.dataframe.loc[self.dataframe['signal']==signal, 'signal'] = 'QQ2HQQ_0J1J'
                self.dataframe.loc[self.dataframe['signal']==signal, 'signal'] = 'QQ2HQQ_0J1J'
            print(1)    
            for signal in ['Diphoton', 'GJet', 'QCD', 'ggH']:
                #self.dataframe.loc[self.dataframe.proc==signal, 'signal'] = signal
                self.dataframe.loc[self.dataframe.proc==signal, 'bkg'] = 1
            print(1)  
            

            

            print('unique signal: ', self.dataframe['signal'].unique())

            print('combinded all data')
        #self.dataframe = self.change_df_dtype(self.dataframe)


    def change_df_dtype(self, df):
        print(df.info())
        cols = df.select_dtypes(include=[np.float64]).columns
        df[cols] = df[cols].astype(np.float32)
        cols = df.select_dtypes(include=[np.int64]).columns
        df[cols] = df[cols].astype(np.int32)
        print(df.info())
        return df
        

    def feature_engineered(self, lep_num = True, PUJID=True):
        if self.loadcomplieddata:
            if lep_num:
                #lep_num 
                self.dataframe['lep_num'] = np.zeros(len(self.dataframe))
                lep_list = ['leadElectronMass', 'leadMuonMass', 'subleadElectronMass', 'subleadMuonMass','subsubleadElectronMass', 'subsubleadMuonMass']
                for index in lep_list:
                    self.dataframe.loc[self.dataframe[index] != -999.0, 'lep_num'] += 1
                print('unique lepton number', self.dataframe['lep_num'].unique())
                self.train_vars += ['lep_num']

            #jet_num 
            self.dataframe['jet_num'] = np.zeros(len(self.dataframe))
            jet_list = ['leadJetEn', 'subleadJetEn', 'subsubleadJetEn']
            for index in jet_list:
                self.dataframe.loc[self.dataframe[index] != -999.0, 'jet_num'] += 1
            print('unique jet number', self.dataframe['jet_num'].unique())
            self.train_vars += ['jet_num']

            #dilepton invariant mass
            #self.dataframe['invariantMass'] = (np.array(self.dataframe.loc[:,'leadElectronEn'])+np.array(self.dataframe.loc[:,'subleadElectronEn']))**2-((np.array(self.dataframe.loc[:,'leadElectronPt'])/np.sin(np.array(self.dataframe.loc[:,'leadElectronEta'])))+(np.array(self.dataframe.loc[:,'subleadElectronPt'])/np.sin(np.array(self.dataframe.loc[:,'subleadElectronEta'])))
            #self.dataframe.loc[self.dataframe['leadElectronCharge']==1, 'leadElectronEn']

            #PUJID
            if PUJID:
                JID_list = ['leadJetPUJID', 'subleadJetPUJID', 'subsubleadJetPUJID']
                ohe = OneHotEncoder(categories='auto', sparse=False)
                ohe_vars = ohe.fit_transform(self.dataframe[JID_list])
                self.onehot_feature = []
                for i in range(0, 15):
                    self.dataframe['PUJID'+str(i)] = ohe_vars[:,i]
                    self.train_vars += ['PUJID'+str(i)]
                    self.onehot_feature.append('PUJID'+str(i))
                del ohe_vars
                gc.collect()
            
            #pthjj
            xt = np.array([])
            yt = np.array([])
            ptfeature = ['leadJetPt','subleadJetPt','leadPhotonPt','subleadPhotonPt']
            etafeature = ['leadJetPhi','subleadJetPhi','leadPhotonPhi','subleadPhotonPhi']
            #self.dataframe['ptHjj'] = np.full(len(self.dataframe), -999.0)
            for i in range(4):
                pt = self.dataframe[ptfeature[i]].replace(-999.0, 0)
                eta = self.dataframe[etafeature[i]].replace(-999.0, 0)
                xt_value = np.array(pt*np.cos(eta)).reshape(1,-1)
                yt_value = np.array(pt*np.sin(eta)).reshape(1,-1)
                if i == 0:
                    xt = xt_value
                    yt = yt_value
                else:
                    xt =  np.concatenate((xt,xt_value),axis=0)
                    yt =  np.concatenate((yt,yt_value),axis=0)
                
            xsum = xt.sum(axis=0)
            ysum = yt.sum(axis=0)
            self.dataframe['ptHjj'] = np.sqrt(xsum**2+ysum**2)
            self.train_vars += ['ptHjj']
            self.train_vars_scale += ['ptHjj']

            
            for var in [xt, yt, xsum, ysum]:
                del var
            gc.collect()
        #self.x_cut = self.dataframe[['dijetMass','ptHjj', 'diphotonPt']]
        self.x_cut = self.dataframe[['dijetMass']]

        
    def create_X_and_y(self, onehoty=True):
        '''
        turn y into one hot encoding;
        and create train test sets for x, y and weights
        '''
        #print(self.dataframe['signal'].unique())
        self.x_train, self.x_test, self.train_weights, self.test_weights, self.train_weights_eq, self.test_weights_eq, self.y_train, self.y_test, self.x_cut_train, self.x_cut_test, self.bkgindex_train, self.bkgindex_test = train_test_split(self.dataframe[self.train_vars], self.dataframe['weight'], 
                                                                                                                                         self.dataframe['weights_eq'], self.dataframe['signal'], self.x_cut, self.dataframe['bkg'],
                                                                                                                                         train_size=0.7,
                                                                                                                                         shuffle=True, 
                                                                                                                                         random_state=21, stratify=self.dataframe['signal']
                                                                                                                                         )


        # encode class values as integers
        if onehoty:
            #print(self.y_train)
            oe = OrdinalEncoder(categories=[self.class_list])
            self.y_train = oe.fit_transform(np.array(self.y_train).reshape(-1,1))
            self.y_test = oe.transform(np.array(self.y_test).reshape(-1,1))

            # convert integers to dummy variables (i.e. one hot encoded)
            self.y_train = keras.utils.np_utils.to_categorical(self.y_train)
            self.y_test = keras.utils.np_utils.to_categorical(self.y_test)
            #print(self.y_train)


    def equalise_weight(self, equalise=True):
        '''
        equalise weights for the differnt signals
        '''
        if self.loadcomplieddata:
            for signal in self.class_list:
                signal_weight = self.dataframe.loc[self.dataframe['signal']==signal, 'weight']
                self.dataframe.loc[self.dataframe['signal']==signal, 'weights_eq'] = signal_weight / sum(signal_weight)
                del signal_weight
                gc.collect()
                

        if equalise:
            self.equalised = True
        else:
            self.equalised = False


    def take_log(self, list):
        '''
        a function that take the log of GeV unit value to scale them down
        '''
        if self.loadcomplieddata:
            for feature in list:
                self.dataframe.loc[self.dataframe[feature]!=-999.0, feature] = np.log(self.dataframe.loc[self.dataframe[feature]!=-999.0, feature])


    def missing_values(self, value=-10.0, scaler=True):
        '''
        assign a value to all missing values
        '''
        if self.loadcomplieddata:
            for feature in ['dijetAbsDEta', 'leadJetEn', 'subleadJetEn', 'subsubleadJetEn', 
            'leadMuonPt', 'leadElectronPt', 'subleadMuonPt', 'subleadElectronPt']:
                try:
                    label = str(feature+'missing')
                    print(label)
                    self.dataframe.loc[:,label] = np.zeros(len(self.dataframe)) 
                    self.dataframe.loc[self.dataframe[feature]==-999.0, label] = 1
                    self.train_vars += [str(label)]
                    gc.collect()
                except KeyError:
                    print('haha')
            
            if scaler:
        
                scale_x = self.dataframe[self.train_vars_scale]
                scale_x = scale_x.replace(-999.0, np.NaN)
                null_values = scale_x.isnull()
                scale_x = scale_x.fillna(value)
                scale_x[~null_values]   = StandardScaler().fit_transform(scale_x[~null_values])
                self.dataframe[self.train_vars_scale] = scale_x
                del scale_x
                gc.collect()
                '''
                self.dataframe[self.train_vars_scale] = self.dataframe[self.train_vars_scale].replace(-999.0, np.NaN)
                gc.collect()
                null_values = self.dataframe[self.train_vars_scale].isnull()
                gc.collect()
                self.dataframe[self.train_vars_scale] = self.dataframe[self.train_vars_scale].fillna(value)
                gc.collect()
                self.dataframe[self.train_vars_scale][~null_values]   = StandardScaler().fit_transform(self.dataframe[self.train_vars_scale][~null_values])
                gc.collect()
                '''
                #print(self.x.head())
                print('scaled the data')
            else:
                self.dataframe = self.dataframe.replace(-999.0, value)



    def train_classifier(self, layers=2, epoch=100, dropout=False, batch_size=64, neurons=10, reg_size=0.001, reg_type='l1', 
    learning_rate=0.001, sch_decay = 0.01, patience = 10, sch_period=10, weight_scaler=10**6, comp_name=None, extra_label='',
    drop_size = 0.2, lstm=False):

        if self.loadcomplieddata == 1:
            self.dataframe.to_csv('Andy/rec_data/datafull_%s.csv' %self.data_label)
            print('saved reconstructed dataframe')

        #self.dataframe = self.change_df_dtype(self.dataframe)
        self.create_X_and_y(onehoty=True)
        scaler = StandardScaler()

        self.x_train[self.train_vars_scale] = scaler.fit_transform(self.x_train[self.train_vars_scale])
        self.x_test[self.train_vars_scale] = scaler.transform(self.x_test[self.train_vars_scale])

        x_train = self.x_train[~self.bkgindex_train]
        x_test = self.x_test[~self.bkgindex_test]

        y_train = self.y_train[~self.bkgindex_train]
        y_test = self.y_test[~self.bkgindex_test]
        if self.equalised:
            train_weights = self.train_weights_eq[~self.bkgindex_train]
            test_weights = self.test_weights_eq[~self.bkgindex_test]
        else:
            train_weights = self.train_weights[~self.bkgindex_train]
            test_weights = self.test_weights[~self.bkgindex_test]


        epoch = epoch
        hidden = layers #number of hidden layers
        dropout = dropout #if add dropout before the first layer
        neurons = neurons #number of neurons each layer
        batch_size = batch_size
        regulation = reg_size
        start_lr = learning_rate
        patience = patience
        exp_decay = sch_decay
        sch_period = sch_period
        drop_size = drop_size
   
        ann_name = ('ann_%slayer_%sneuron_%sreg%s_ls%s_decay%s_%s' %(layers,neurons,reg_type,reg_size,learning_rate,sch_decay,extra_label)).replace('0.', '')
        if dropout:
            ann_name += '_dropout'
            ann_name += str(drop_size).replace('0.','')
        self.ann_name = ann_name
        compare_name = comp_name
        self.compare_name = comp_name

        #scaler for weights
        weight_scaler = weight_scaler

        print('Name of classifier: %s' %ann_name)


        # Define the scheduling function
        def schedule(epoch):
            def lr(epoch, start_lr, exp_decay):
                return start_lr * math.exp(-exp_decay*epoch)
            
            if epoch < sch_period:
                return start_lr
            elif epoch < sch_period*2:
                return lr(sch_period*2, start_lr, exp_decay)
            elif epoch < sch_period*3:
                return lr(sch_period*3, start_lr, exp_decay)
            elif epoch < sch_period*4:
                return lr(sch_period*4, start_lr, exp_decay)
            else:
                return lr(epoch, start_lr, exp_decay)

        try: 
            model = pickle.load(open("Andy/model_multi/%s/%s.pickle.dat" %(self.folder,ann_name)))
            print('sucessufuly loaded ANN model')
            train_model = 0

        except IOError:
            train_model = 1

            model = Sequential()
            model.add(Dense(len(self.train_vars), activation='relu', input_dim = len(self.train_vars)))

            for i in range(hidden):
                print('add 1 hidden layer')
                model.add(Dense(neurons, activation='relu',activity_regularizer=keras.regularizers.l1(regulation)))
                if dropout:
                    model.add(Dropout(drop_size))

            model.add(Dense(len(self.class_list), activation='softmax'))
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
            #optimizer = keras.optimizers.Adam(lr = learning_rate)
            optimizer = keras.optimizers.Nadam(lr = learning_rate)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


            print(model.summary())
            print('training a BDT')
            print('input features: \n%s'%self.train_vars)
            hist = model.fit(x_train,y_train,sample_weight=train_weights.to_numpy()*weight_scaler, batch_size=batch_size, epochs=epoch, verbose=1,
            validation_data=(x_test, y_test,test_weights.to_numpy()*weight_scaler),
            callbacks = [keras.callbacks.LearningRateScheduler(schedule, verbose=1), early_stopping],
            )
            pickle.dump(model, open("Andy/model_multi/%s/%s.pickle.dat" %(self.folder,ann_name),  "wb"))

        self.classifer = model


        #get prediction for sig and background
        self.y_pred_test = model.predict_proba(self.x_test)
        self.y_pred_train = model.predict_proba(self.x_train)

        #get scores and weights for each individual signal
        self.scores_dic = {}
        self.weight_dic = {}
        self.class_train_index = {}
        self.class_test_index = {}
        for i, signal in enumerate(self.class_list):
            self.scores_dic[signal] = self.y_pred_test [(y_test[:,i]==1)]
            self.weight_dic[signal] = self.test_weights [(y_test[:,i]==1)]
            self.class_train_index[signal] = self.y_pred_train[len(self.y_pred_train),self.y_pred_train.argmax(axis=1)]
            self.class_test_index[signal] = self.y_pred_test[len(self.y_pred_test),self.y_pred_test.argmax(axis=1)]

        #plot training history
        if train_model == 0:
            history_ann = pd.read_csv('Andy/train_history/%s/%s.csv' %(self.folder,ann_name))
        else:
            history_ann = hist.history
            #save history
            hist_df = pd.DataFrame(hist.history)
            hist_df.to_csv('Andy/train_history/%s/%s.csv' %(self.folder,ann_name))

        plt.plot(history_ann['loss'], label='loss %s'%ann_name.replace('_',' '), color='blue')
        plt.plot(history_ann['val_loss'], label='val loss %s'%ann_name.replace('_',' '), color='orange')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')


        plt.legend(loc='upper right') 
        plt.savefig('plotting/plots/%s/training_history_%s' %(self.folder,ann_name))

        plt.close()

        print('accuracy: ', accuracy_score(self.y_test.argmax(axis=1), self.y_pred_test.argmax(axis=1), sample_weight=self.test_weights))
        print('accuracy eq: ', accuracy_score(self.y_test.argmax(axis=1), self.y_pred_test.argmax(axis=1), sample_weight=self.test_weights_eq))
        
        #print('accuracy: ', accuracy_score(self.y_test.argmax(axis=1)[self.y_test], self.y_pred_test.argmax(axis=1), sample_weight=self.test_weights))


    def train_bdt(self, bdtname=''):
        self.create_X_and_y(onehoty=False)
        x_train = self.x_train
        x_test = self.x_test
        #self.x_train[self.train_vars_scale] = scaler.fit_transform(self.x_train[self.train_vars_scale])
        #self.x_test[self.train_vars_scale] = scaler.transform(self.x_test[self.train_vars_scale])

        self.ann_name = bdtname


        if self.equalised:
            train_weights = self.train_weights_eq
            test_weights = self.test_weights_eq
        else:
            train_weights = self.train_weights
            test_weights = self.test_weights

        y_train = self.y_train
       
        try: 
            model = pickle.load(open('Andy/model_multi/%s/%s.pickle.dat'%(self.folder,bdtname)))
            print('sucessufuly loaded model')
        except IOError:
            print('training a BDT')
            model = xgb.XGBClassifier(objective='multi:softprob',
                #n_estimators=200, 
                #learning_rate=0.25, 
                #max_depth=10, 
                #subsample=0.2
                )
            model.fit(x_train,y_train,train_weights)
            pickle.dump(model, open("Andy/model_multi/%s/%s.pickle.dat"%(self.folder,bdtname),  "wb"))
        
        self.class_list = model.classes_
        self.classifer = model

        oe = OrdinalEncoder(categories=[self.class_list])
        self.y_test = oe.fit_transform(np.array(self.y_test).reshape(-1,1))
        self.y_test = keras.utils.np_utils.to_categorical(self.y_test)

        y_test = self.y_test

        #get prediction for sig and background
        self.y_pred_test = model.predict_proba(x_test)
        self.y_pred_train = model.predict_proba(x_train)
 
        #get scores and weights for each individual signal
        self.scores_dic = {}
        self.weight_dic = {}
        #self.class_train_index = {}
        for i, signal in enumerate(self.class_list):
            self.scores_dic[signal] = self.y_pred_test [(y_test[:,i]==1)]
            self.weight_dic[signal] = test_weights [(y_test[:,i]==1)]
            #self.class_train_index[signal] = self.y_pred_train.argmax(axis=1)

        print('accuracy: ', accuracy_score(self.y_test.argmax(axis=1), self.y_pred_test.argmax(axis=1), sample_weight=self.test_weights))
        print('accuracy eq: ', accuracy_score(self.y_test.argmax(axis=1), self.y_pred_test.argmax(axis=1), sample_weight=self.test_weights_eq))


    def train_binary(self):
        for i, signal in enumerate(self.class_list):
            x = self.class_train_index[i]


    def plot_roc(self, y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = False, label = '', color='red', title=None):
        '''
        built-in function
        '''
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
        axes.set_title(title)
        axes.legend(ncol=1, prop={'size':10}, loc='upper right')
        axes.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)


    def calculate_auc(self, bkg_eff, sig_eff):
        '''
        build-in function
        '''
        auc = integrate.trapz(x=bkg_eff, y=sig_eff)
        return auc 


    def plot_output(self, bins=np.linspace(0,1,31)):
        for i, score in enumerate(self.class_list):
            fig  = plt.figure(1)
            axes = fig.gca()       
            bins = bins
            axes.set_title(self.ann_name.replace('_', ' '))

            for signal in self.class_list:
                axes.hist(self.scores_dic[signal][:,i], bins=bins, label=signal.replace('_',' '), weights=self.weight_dic[signal], histtype='step', color = self.color_map[signal])
 

            axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
            axes.set_xlabel('%s Score' %score.replace('_',' '), ha='right', x=1, size=13)
            current_bottom, current_top = axes.get_ylim()
            axes.set_yscale('log', nonposy='clip')
            axes.set_ylim(top=current_top*2)
            axes.legend(ncol=2, prop={'size':5})

            
            plt.savefig('plotting/plots/%s/output_score_%s_%s.png' %(self.folder,score,self.ann_name))
            plt.close()


    def output_class_stack(self):


        '''
        self.y_pred_test_argmax = self.y_pred_test.argmax(axis=1)
        self.scores_dic_argmax = {}
        self.weight_dic_eq = {}
        for i, signal in enumerate(self.class_list):
            self.scores_dic_argmax[signal] = self.y_pred_test_argmax [(self.y_test.argmax(axis=1)==i)]
            self.weight_dic_eq[signal] =  self.test_weights_eq [(self.y_test[:,i]==1)]
        '''

        stacked_color = []
        stacked_label = []
        stacked = []
        stacked_weight1 = []       
        stacked_weight2 = []
        for i, signal in enumerate(self.class_list):
            stacked.append(np.arange(0,len(self.class_list)))

            stacked_weight1.append(self.array1[:,i])
            stacked_weight2.append(self.array2[i,:])

            stacked_color.append(self.color_map[signal])
            stacked_label.append(signal.replace('_',' '))

        j = 1
        for stack in [stacked_weight1, stacked_weight2]:
            fig  = plt.figure(1)
            axes = fig.gca()       
            bins=np.arange(0, len(self.class_list)+1, 1)

            axes.hist(stacked,align='mid',rwidth=0.8, density=True, stacked=True, bins=bins, label=stacked_label, weights=stack, histtype='barstacked',)

            axes.set_ylabel('Normalized Units', ha='right', y=1, size=13)
            
            #axes.set_xlim(-0.5,6.5)
            axes.legend(prop={'size':10})
            current_bottom, current_top = axes.get_ylim()
            axes.set_ylim(top=current_top*1.5)

            if j ==1:
                axes.set_title('Nomalised by Prediction')
                axes.set_xlabel('Prediction Classes')
            if j==2:
                axes.set_title('Nomalised by True')
                axes.set_xlabel('True Classes')

            plt.savefig('plotting/plots/%s/stacked_%s_%s.png' %(self.folder,j,self.ann_name))
            plt.close()
            j += 1
    

    def plot_roc_curves(self):
        for i, score in enumerate(self.class_list):
            self.plot_roc(y_train=self.y_train[:,i], y_pred_train=self.y_pred_train[:,i], train_weights=self.train_weights, y_test=self.y_test[:,i], y_pred_test=self.y_pred_test[:,i], test_weights=self.test_weights, label=self.ann_name.replace('_', ' '), color='blue',title='%s roc curve' %score.replace('_',' '))
 
            #if self.compare_name != None:
            #    self.plot_roc(self.y_train[:,i], self.y_pred_train_comp, self.train_weights, self.y_test[:,i], self.y_pred_test_comp[:,i], self.test_weights, label=self.compare_name.replace('_', ' '), color='red',title='%s roc curve' %score.replace('_',' '))
            #    plt.savefig('plotting/plots/%s/roc_%s_%s_against_%s.png'%(self.folder,score,self.ann_name,self.compare_name))
        
            plt.savefig('plotting/plots/%s/roc_%s_%s.png'%(self.folder,score,self.ann_name))
            plt.close()

            bkg_eff_test, sig_eff_test, _ = roc_curve(self.y_test[:,i], self.y_pred_test[:,i], sample_weight=self.test_weights)
            #bkg_eps = bkg_eff_test[find_eff(sig_eff_test, 0.9)]
            #print('bkg efficiency with signal eff 0.9: %s' %round(bkg_eps,4))
            print('auc for %s against the rest: %s' %(score,round(self.calculate_auc(bkg_eff_test, sig_eff_test),4)))
        
        
    def plot_confusion_matrix(self, y_pred, label=''):
        fig  = plt.figure(1)
        axes = fig.gca()     
        array1 = confusion_matrix(y_pred.argmax(axis=1), self.y_test.argmax(axis=1), sample_weight=self.test_weights)
        array2 = array1.copy()
        array3 = array1.copy()
        #print('experimental accuracy:', np.sum([array[i,i] for i in range(np.shape(array1)[0]-1)])/np.sum(array1[0:-2,:]))

        j = 1
        for array in [array1, array2]:
            if array is array1:
                array = array.astype('float') / array.sum(axis=2-j)[:, np.newaxis]
                self.array1 = array
            elif array is array2:
                array = array.astype('float') / array.sum(axis=2-j)
                self.array2 = array

            array = np.round(array, 2)
            df_cm = pd.DataFrame(array, index = [i.replace('_', ' ') for i in self.class_list],
                    columns = [i.replace('_', ' ') for i in self.class_list])
            cmap = sn.diverging_palette(230, 20, as_cmap=True)
            axes = sn.heatmap(df_cm, annot=True, cmap=cmap, vmax=1.0, center=0, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 9})
            axes.invert_yaxis()
            axes.set_ylabel('Predicted')
            axes.set_xlabel('True')
            axes.set_title('Confusion Matrix %s for %s' %(label, self.ann_name.replace('_',' ')))

            plt.savefig('plotting/plots/%s/confusion%s_%s_%s.png' %(self.folder,j,label,self.ann_name))      
            plt.close() 
            j += 1 

        #array3
        array3 = np.abs(array3)
        #array3 = np.log(array3)
        array3 = array3 - np.amin(array3)
        array4 = array3[1:,1:] #exclude rest class
        j = 3
        for array in [array3, array4]:
            if array is array3:
                df_cm = pd.DataFrame(array, index = [i.replace('_', ' ') for i in self.class_list],
                        columns = [i.replace('_', ' ') for i in self.class_list])
            if array is array4:
                df_cm = pd.DataFrame(array, index = [i.replace('_', ' ') for i in self.class_list[1:]],
                        columns = [i.replace('_', ' ') for i in self.class_list[1:]])
            #cmap = sn.color_palette('YlOrBr')
            cmap = sn.diverging_palette(230, 20, as_cmap=True)
            axes = sn.heatmap(df_cm, annot=True, cmap=cmap, center=0, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 9})
            axes.invert_yaxis()
            axes.set_ylabel('Predicted')
            axes.set_xlabel('True')
            axes.set_title('Absolute Confusion Matrix %s for %s' %(label, self.ann_name.replace('_',' ')))

            plt.savefig('plotting/plots/%s/confusion%s_%s_%s.png' %(self.folder,j,label,self.ann_name))      
            plt.close() 
            j += 1


    def stxs_cut(self):
        y_pred = pd.DataFrame(np.zeros([len(self.x_test),len(self.class_list)]), columns=self.class_list)
        self.x_cut_test = self.x_cut_test.reset_index(drop=True)
        self.x_test = self.x_test.reset_index(drop=True)
        y_pred.loc[self.x_test['jet_num']==0, 'QQ2HQQ_0J1J'] = 1
        y_pred.loc[self.x_test['jet_num']==1, 'QQ2HQQ_0J1J'] = 1
        y_pred.loc[(self.x_test['jet_num']>=2)&(self.x_cut_test['dijetMass']>0)&(self.x_cut_test['dijetMass']<60), 'QQ2HQQ_0J1J'] = 1
        y_pred.loc[(self.x_test['jet_num']>=2)&(self.x_cut_test['dijetMass']>120)&(self.x_cut_test['dijetMass']<350), 'QQ2HQQ_0J1J'] = 1
        y_pred.loc[(self.x_test['jet_num']>=2)&(self.x_cut_test['dijetMass']>60)&(self.x_cut_test['dijetMass']<120), 'QQ2HQQ_GE2J_MJJ_60_120'] = 1
        y_pred.loc[(self.x_test['jet_num']>=2)&(self.x_cut_test['dijetMass']>350)&(self.x_cut_test['dijetMass']<700)&(self.x_cut_test['ptHjj']>0)&(self.x_cut_test['ptHjj']<25), 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25'] = 1
        y_pred.loc[(self.x_test['jet_num']>=2)&(self.x_cut_test['dijetMass']>700)&(self.x_cut_test['ptHjj']>0)&(self.x_cut_test['ptHjj']<25), 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25'] = 1
        y_pred.loc[(self.x_test['jet_num']>=2)&(self.x_cut_test['dijetMass']>350)&(self.x_cut_test['dijetMass']<700)&(self.x_cut_test['ptHjj']>25), 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25'] = 1
        y_pred.loc[(self.x_test['jet_num']>=2)&(self.x_cut_test['dijetMass']>700)&(self.x_cut_test['ptHjj']>25), 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25'] = 1
        y_pred.loc[(self.x_test['jet_num']>=2)&(self.x_cut_test['dijetMass']>350)&(self.x_cut_test['diphotonPt']>200), 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200'] = 1

        self.y_pred_test_cut = np.array(y_pred)
        print('cut accuracy: ', accuracy_score(self.y_test.argmax(axis=1), self.y_pred_test_cut.argmax(axis=1), sample_weight=self.test_weights))




