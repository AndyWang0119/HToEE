import argparse
from locale import normalize
from unicodedata import category
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from xgboost import plot_importance
import xgboost as xgb
import pickle
from keras import Sequential
from keras.layers import Dense, Dropout
import keras     
from scipy import integrate
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import math
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
plt.style.use("cms10_6_HP")
import matplotlib as mpl
import seaborn as sn
#loading data
parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('Required Arguments')
required_args.add_argument('-d','--data', action='store', help='Input data dir', required=True)
options=parser.parse_args()

dataframe_ggh = pd.read_csv('%s/ggH_multiclass_ann_df_2017.csv'%options.data)
dataframe_vbf = pd.read_csv('%s/VBF_multiclass_ann_df_2017.csv'%options.data)
dataframe_vh = pd.read_csv('%s/VH_multiclass_ann_df_2017.csv'%options.data)
dataframe_tth = pd.read_csv('%s/ttH_multiclass_ann_df_2017.csv'%options.data)
dataframe_thw = pd.read_csv('%s/tHW_multiclass_ann_df_2017.csv'%options.data)
dataframe_thq = pd.read_csv('%s/tHq_multiclass_ann_df_2017.csv'%options.data)
print('Loaded all data')



class multiClass(object):
    def __init__(self, dataframe_ggh, dataframe_vbf, dataframe_vh, dataframe_tth, dataframe_thw, dataframe_thq, train_vars, 
    folder, class_list, stxs_list, color_map):
        '''
        :param class_list: class labels that want the classifier to classify
        :type stxs_list: list
        :param stxs_list: actual HTXS_1.2 label from the ROOT file
        :type stxs_list: list
        '''
        self.dataframe_ggh = dataframe_ggh
        self.dataframe_vbf = dataframe_vbf
        self.dataframe_vh = dataframe_vh
        self.dataframe_tth = dataframe_tth
        self.dataframe_thw = dataframe_thw
        self.dataframe_thq = dataframe_thq
        self.folder = folder
        self.class_list = class_list
        self.stxs_list = stxs_list
        self.color_map = color_map

        self.train_vars = train_vars

        self.classifer = None
        self.classifer_comp = None

        self.equalised = False
    
    def combine_dataframe(self):
        '''
        assign signal type to the dataframe;
        combine them into a single dataframe;
        '''
        for df in [self.dataframe_ggh, self.dataframe_vbf, self.dataframe_vh, self.dataframe_tth, self.dataframe_thw, self.dataframe_thq]:
            df['signal'] = np.zeros(len(df))
        #self.dataframe_ggh['signal'] = np.zeros(len(self.dataframe_ggh))
        #self.dataframe_vbf['signal'] = np.zeros(len(self.dataframe_vbf))
        #self.dataframe_vh['signal'] = np.zeros(len(self.dataframe_vh))
        #self.dataframe_tth['signal'] = np.zeros(len(self.dataframe_tth))
        #self.dataframe_thw['signal'] = np.zeros(len(self.dataframe_thw))
        #self.dataframe_thq['signal'] = np.zeros(len(self.dataframe_thq))

        for index in [10,11]: #ggH
            self.dataframe_ggh.loc[self.dataframe_ggh['HTXS_stage_0']==index, 'signal'] = 'ggH'
        for index in [20,21,22,23,]: #qqH
            self.dataframe_vbf.loc[self.dataframe_vbf['HTXS_stage_0']==index, 'signal'] = self.dataframe_vbf.loc[self.dataframe_vbf['HTXS_stage_0']==index, 'signal'].replace(0, 'qqH')
            self.dataframe_vh.loc[self.dataframe_vh['HTXS_stage_0']==index, 'signal'] = self.dataframe_vh.loc[self.dataframe_vh['HTXS_stage_0']==index, 'signal'].replace(0, 'qqH')
        for index in [30,31]: #WH
            self.dataframe_vh.loc[self.dataframe_vh['HTXS_stage_0']==index, 'signal'] = 'WH'
        for index in [40,41]: #ZH
            self.dataframe_vh.loc[self.dataframe_vh['HTXS_stage_0']==index, 'signal'] = 'ZH'
        for index in [60,61]: #ttH
            self.dataframe_tth.loc[self.dataframe_tth['HTXS_stage_0']==index, 'signal'] = 'ttH'
        for index in [80,81]: #tH
            self.dataframe_thw.loc[self.dataframe_thw['HTXS_stage_0']==index, 'signal'] = 'tH'
            self.dataframe_thq.loc[self.dataframe_thq['HTXS_stage_0']==index, 'signal'] = 'tH'
        
     
        for i, signal in enumerate(self.stxs_list):
            i += 200
            self.dataframe_vbf.loc[self.dataframe_vbf['HTXS_stage1_2_cat_pTjet30GeV']==i, 'signal'] = signal
            self.dataframe_vh.loc[self.dataframe_vh['HTXS_stage1_2_cat_pTjet30GeV']==i, 'signal'] = signal

        dataframe = pd.concat([self.dataframe_ggh, self.dataframe_vbf, self.dataframe_vh, self.dataframe_tth,dataframe_thw,dataframe_thq])
        
        #conbine the 0 1 jet stxs bins
        for signal in ['QQ2HQQ_0J', 'QQ2HQQ_1J', 'QQ2HQQ_GE2J_MJJ_0_60', 'QQ2HQQ_GE2J_MJJ_120_350']:
            dataframe.loc[dataframe['HTXS_stage1_2_cat_pTjet30GeV']==signal, 'signal'] = self.class_list[0]

        #remove those signal that is not in the class list
        for i in dataframe['signal'].unique():
            if i not in self.class_list:
                print(i)
                dataframe = dataframe.loc[dataframe['signal'] != i]

        self.dataframe = dataframe
        print('combinded all data')

    def feature_engineered(self):
        #lep_num 
        self.dataframe['lep_num'] = np.zeros(len(self.dataframe))
        lep_list = ['leadElectronMass', 'leadMuonMass', 'subleadElectronMass', 'subleadMuonMass','subsubleadElectronMass', 'subsubleadMuonMass']
        for index in lep_list:
            self.dataframe.loc[self.dataframe[index] != -999.0, 'lep_num'] += 1
        
        self.train_vars += ['lep_num']

        #dilepton invariant mass


        #PUJID
        JID_list = ['leadJetPUJID', 'subleadJetPUJID', 'subsubleadJetPUJID']
        ohe = OneHotEncoder(categories='auto', sparse=False)
        ohe_vars = ohe.fit_transform(self.dataframe[JID_list])
        for i in range(0, 15):
            self.dataframe['PUJID'+str(i)] = ohe_vars[:,i]
            self.train_vars += ['PUJID'+str(i)]


    def create_X_and_y(self, onehoty=True):
        '''
        turn y into one hot encoding;
        and create train test sets for x, y and weights
        '''
        print(self.dataframe['signal'].unique())
        self.x_train, self.x_test, self.train_weights, self.test_weights, self.train_weights_eq, self.test_weights_eq, self.y_train, self.y_test= train_test_split(self.dataframe[self.train_vars], self.dataframe['weight'], 
                                                                                                                                         self.dataframe['weights_eq'], self.dataframe['signal'], 
                                                                                                                                         train_size=0.7,
                                                                                                                                         shuffle=True, 
                                                                                                                                         random_state=1357
                                                                                                                                         )


        # encode class values as integers
        if onehoty:
            print(self.y_train)
            oe = OrdinalEncoder(categories=[self.class_list])
            encoded_train = oe.fit_transform(np.array(self.y_train).reshape(-1,1))
            encoded_test = oe.transform(np.array(self.y_test).reshape(-1,1))

            # convert integers to dummy variables (i.e. one hot encoded)
            self.y_train = keras.utils.np_utils.to_categorical(encoded_train)
            self.y_test = keras.utils.np_utils.to_categorical(encoded_test)
            print(self.y_train)



    def equalise_weight(self, equalise=True):
        '''
        equalise weights for the differnt signals
        '''
        for signal in self.class_list:
            signal_weight = self.dataframe.loc[self.dataframe['signal']==signal, 'weight']
            self.dataframe.loc[self.dataframe['signal']==signal, 'weights_eq'] = signal_weight / sum(signal_weight)

        if equalise:
            self.equalised = True
        else:
            self.equalised = False


    def take_log(self, list):
        '''
        a function that take the log of GeV unit value to scale them down
        '''
        for feature in list:
            self.dataframe.loc[self.dataframe[feature]!=-999.0, feature] = np.log(self.dataframe.loc[self.dataframe[feature]!=-999.0, feature])


    def missing_values(self, value=-10.0):
        '''
        assign a value to all missing values
        '''
        self.dataframe = self.dataframe.replace(-999.0, value)

    def train_bdt(self, bdtname=''):
        self.create_X_and_y(onehoty=False)
        #scaler = StandardScaler()
        #self.x_train = scaler.fit_transform(self.x_train)
        #self.x_test = scaler.transform(self.x_test)
        if self.equalised:
            train_weights = self.train_weights_eq
            test_weights = self.test_weights_eq
        else:
            train_weights = self.train_weights
            test_weights = self.test_weights

        try: 
            self.classifer = pickle.load(open('Andy/model_multi/%s/bdt_%s.pickle.dat'%(self.folder, bdtname)))
            print('sucessufuly loaded model')
        except IOError:
            print('training a BDT')
            self.classifer = xgb.XGBClassifier(objective='multi:softprob', 
                    num_classes=len(self.class_list),
                    n_estimators=200, 
                    learning_rate=0.25, 
                    max_depth=10, 
                    subsample=0.2
                    ) 
            self.classifer.fit(self.x_train,self.y_train,train_weights)
            pickle.dump(self.classifer, open('Andy/model_multi/%s/bdt_%s.pickle.dat'%(self.folder, bdtname),  "wb"))


        #get prediction for sig and background
        self.y_pred_test = self.classifer.predict_proba(self.x_test)
        self.y_pred_train = self.classifer.predict_proba(self.x_train)
        print(self.y_pred_test)
        #get scores and weights for each individual signal
        self.scores_dic = {}
        self.weight_dic = {}
        for i, signal in enumerate(self.class_list):
            self.scores_dic[signal] = self.y_pred_test [(self.y_test==signal)]
            self.weight_dic[signal] = test_weights [(self.y_test==signal)]



    def train_classifier(self, layers=2, epoch=100, dropout=False, batch_size=64, neurons=10, reg_size=0.001, reg_type='l1', 
    learning_rate=0.001, sch_decay = 0.01, patience = 10, weight_scaler=10**6, comp_name=None, extra_label=''):

        self.create_X_and_y(onehoty=True)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(self.x_train)
        x_test = scaler.transform(self.x_test)
        y_train = self.y_train
        y_test = self.y_test
        if self.equalised:
            train_weights = self.train_weights_eq
            test_weights = self.test_weights_eq
        else:
            train_weights = self.train_weights
            test_weights = self.test_weights


        epoch = epoch
        hidden = layers #number of hidden layers
        dropout = dropout #if add dropout before the first layer
        neurons = neurons #number of neurons each layer
        batch_size = batch_size
        regulation = reg_size
        start_lr = learning_rate
        patience = patience
        exp_decay = sch_decay
   
        ann_name = ('ann_%slayer_%sneuron_%sreg%s_ls%s_decay%s_%s' %(layers,neurons,reg_type,reg_size,learning_rate,sch_decay,extra_label)).replace('0.', '')
        if dropout:
            ann_name += '_dropout'
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
            if epoch < 10:
                return start_lr
            elif epoch < 20:
                return lr(20, start_lr, exp_decay)
            elif epoch < 30:
                return lr(30, start_lr, exp_decay)
            elif epoch < 40:
                return lr(40, start_lr, exp_decay)
            else:
                return lr(epoch, start_lr, exp_decay)

        try: 
            model = pickle.load(open("Andy/model_multi/%s/%s.pickle.dat" %(self.folder,ann_name)))
            print('sucessufuly loaded ANN model')
            train_model = 0

        except IOError:
            train_model = 1
            model = Sequential()
            model.add(Dense(len(features), activation='relu', input_dim = len(features)))

            for i in range(hidden):
                print('add 1 hidden layer')
                model.add(Dense(neurons, activation='relu',activity_regularizer=keras.regularizers.l1(regulation)))
                if dropout:
                    model.add(Dropout(0.2))

            model.add(Dense(len(self.class_list), activation='softmax'))
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
            #optimizer = keras.optimizers.Adam(lr = 0.001)
            optimizer = keras.optimizers.Nadam(lr = learning_rate)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy')
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
        self.y_pred_test = model.predict_proba(x_test)
        self.y_pred_train = model.predict_proba(x_train)

        #get scores and weights for each individual signal
        self.scores_dic = {}
        self.weight_dic = {}
        for i, signal in enumerate(self.class_list):
            self.scores_dic[signal] = self.y_pred_test [(y_test[:,i]==1)]
            self.weight_dic[signal] = test_weights [(y_test[:,i]==1)]
        #self.ggh_scores = self.y_pred_test [(y_test[:,0]==1)]
        #self.qqh_scores = self.y_pred_test [(y_test[:,1]==1)]
        #self.wh_scores = self.y_pred_test [(y_test[:,2]==1)]
        #self.zh_scores = self.y_pred_test [(y_test[:,3]==1)]
        #self.tth_scores = self.y_pred_test [(y_test[:,4]==1)]
        #self.th_scores = self.y_pred_test [(y_test[:,5]==1)]

        #self.ggh_weights = test_weights [(y_test[:,0]==1)]
        #self.qqh_weights = test_weights [(y_test[:,1]==1)]
        #self.zh_weights = test_weights [(y_test[:,2]==1)]
        #self.wh_weights = test_weights [(y_test[:,3]==1)]
        #self.tth_weights = test_weights [(y_test[:,4]==1)]
        #self.th_weights = test_weights [(y_test[:,5]==1)]
        

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


    def plot_output(self, bins=np.linspace(0,1,41), argmax=False):
        for i, score in enumerate(self.class_list):
            fig  = plt.figure(1)
            axes = fig.gca()       
            bins = bins
            axes.set_title(self.ann_name.replace('_', ' '))

            for signal in self.class_list:
                axes.hist(self.scores_dic[signal][:,i], bins=bins, label=signal.replace('_',' '), weights=self.weight_dic[signal], histtype='step', color = self.color_map[signal])
                if argmax:
                    arg_label = 'argmax'
                    axes.hist(self.scores_dic.argmax(axis=1)[signal][:,i], bins=bins, label=signal.replace('_',' '), weights=self.weight_dic[signal], histtype='step', color = self.color_map[signal])
                else:
                    arg_label = ''
                    axes.hist(self.scores_dic[signal][:,i], bins=bins, label=signal.replace('_',' '), weights=self.weight_dic[signal], histtype='step', color = self.color_map[signal])


            axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
            axes.set_xlabel('%s Score' %score.replace('_',' '), ha='right', x=1, size=13)
            current_bottom, current_top = axes.get_ylim()
            axes.set_yscale('log', nonposy='clip')
            axes.set_ylim(top=current_top*2)

            
            plt.savefig('plotting/plots/%s/%s_output_score_%s_%s.png' %(self.folder,arg_label,score,self.ann_name))
            plt.close()


    def plot_roc_curves(self):
        for i, score in enumerate(self.class_list):
            self.plot_roc(y_train=self.y_train[:,i], y_pred_train=self.y_pred_train[:,i], train_weights=self.train_weights, y_test=self.y_test[:,i], y_pred_test=self.y_pred_test[:,i], test_weights=self.test_weights, label=self.ann_name.replace('_', ' '), color='blue',title='%s roc curve' %score.replace('_',' '))
 
            if self.compare_name != None:
                self.plot_roc(self.y_train[:,i], self.y_pred_train_comp, self.train_weights, self.y_test[:,i], self.y_pred_test_comp[:,i], self.test_weights, label=self.compare_name.replace('_', ' '), color='red',title='%s roc curve' %score.replace('_',' '))
                plt.savefig('plotting/plots/%s/roc_%s_%s_against_%s.png'%(self.folder,score,self.ann_name,self.compare_name))
            else:
                plt.savefig('plotting/plots/%s/roc_%s_%s.png'%(self.folder,score,self.ann_name))
            plt.close()

            bkg_eff_test, sig_eff_test, _ = roc_curve(self.y_test[:,i], self.y_pred_test[:,i], sample_weight=self.test_weights)
            #bkg_eps = bkg_eff_test[find_eff(sig_eff_test, 0.9)]
            #print('bkg efficiency with signal eff 0.9: %s' %round(bkg_eps,4))
            print('auc for %s against the rest: %s' %(score,round(self.calculate_auc(bkg_eff_test, sig_eff_test),4)))

    def plot_confusion_matrix(self):
        fig  = plt.figure(1)
        axes = fig.gca()     
        print('accuracy: ', accuracy_score(self.y_test.argmax(axis=1), self.y_pred_test.argmax(axis=1), sample_weight=self.test_weights))
        print('accuracy eq: ', accuracy_score(self.y_test.argmax(axis=1), self.y_pred_test.argmax(axis=1), sample_weight=self.test_weights_eq))
        array = confusion_matrix(self.y_test.argmax(axis=1), self.y_pred_test.argmax(axis=1), sample_weight=self.test_weights)
        #print(array)
        #array =  array / array.astype(np.float).sum(axis=1)
        array = array.astype('float') / array.sum(axis=1)[:, np.newaxis]
        array = np.round(array, 2)
        df_cm = pd.DataFrame(array, index = [i.replace('_', ' ') for i in self.class_list],
                 columns = [i.replace('_', ' ') for i in self.class_list])
        cmap = sn.diverging_palette(230, 20, as_cmap=True)
        axes = sn.heatmap(df_cm, annot=True, cmap=cmap, vmax=1.0, center=0, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 9})
        axes.invert_yaxis()
        axes.set_ylabel('True')
        axes.set_xlabel('Predicted')

        plt.savefig('plotting/plots/%s/confusion_%s.png' %(self.folder,self.ann_name))      
        plt.close()  
        


features =        ['diphotonPt','diphotonCosPhi', 'leadPhotonPtOvM', 'subleadPhotonPtOvM', 'leadPhotonEta', 'subleadPhotonEta',
    'dijetAbsDEta', 'dijetDPhi','dijetMinDRJetPho', 'dijetMass','dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'dijetCentrality',
    'leadJetDiphoDPhi', 'subleadJetDiphoDPhi', 'leadJetDiphoDEta', 'subleadJetDiphoDEta', 
    #'subsubleadJetDiphoDPhi', 'subsubleadJetDiphoDEta',
    'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL','subleadJetEn', 'subleadJetPt', 'subleadJetEta', 'subleadJetPhi','subleadJetQGL',
    'subsubleadJetEn', 'subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetPhi','subsubleadJetQGL',

   'leadJetBTagScore','subleadJetBTagScore', 'subsubleadJetBTagScore',
    #'leadJetPUJID', 'subleadJetPUJID', 'subsubleadJetPUJID', 
    'nSoftJets', 'metPt', 'metPhi', 'metSumET', 'metSignificance',
    #'leadElectronMass', 'leadMuonMass', 'subleadElectronMass', 'subleadMuonMass','subsubleadElectronMass', 'subsubleadMuonMass',
    'leadElectronEn', 'leadMuonEn', 'leadElectronPt', 'leadMuonPt', 'leadElectronEta', 'leadMuonEta', 'leadElectronPhi', 'leadMuonPhi',
    'leadElectronCharge', 'leadElectronConvVeto', 'leadMuonCharge',
    'subleadElectronEn', 'subleadMuonEn', 'subleadElectronPt', 'subleadMuonPt', 'subleadElectronEta', 'subleadMuonEta', 'subleadElectronPhi', 'subleadMuonPhi',
    'subleadElectronCharge', 'subleadElectronConvVeto', 'subleadMuonCharge',
    'subsubleadElectronEn', 'subsubleadMuonEn', 'subsubleadElectronPt', 'subsubleadMuonPt', 'subsubleadElectronEta', 'subsubleadMuonEta', 'subsubleadElectronPhi', 'subsubleadMuonPhi',
    'subsubleadElectronCharge', 'subsubleadElectronConvVeto', 'subsubleadMuonCharge',
    ]

Gev_list = ['diphotonPt', 'dijetMass', 'leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt','metPt', 'subsubleadJetEn', 'subsubleadJetPt',
            #'leadElectronMass', 'leadMuonMass', 'subleadElectronMass', 'subleadMuonMass','subsubleadElectronMass', 'subsubleadMuonMass'
            ]

class_list = ['ggH', 'qqH', 'WH', 'ZH', 'ttH', 'tH']
class_list2 = [
#'ggH', 'WH', 'ZH', 'ttH', 'tH',
'QQ2HQQ_0J1J',
'QQ2HQQ_GE2J_MJJ_60_120', 
'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25', 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']

stxs_list = ['QQ2HQQ_0J', 'QQ2HQQ_1J', 'QQ2HQQ_GE2J_MJJ_0_60', 'QQ2HQQ_GE2J_MJJ_60_120', 'QQ2HQQ_GE2J_MJJ_120_350',
'QQ2HQQ_FWDH','QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25', 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']

color_map2 = {'ggH':'blue', 'WH':'green','ZH':'springgreen','ttH':'purple','tH':'yellow',
'QQ2HQQ_FWDH':'bisque','QQ2HQQ_0J':'darkorange', 'QQ2HQQ_1J':'burlywood', 'QQ2HQQ_GE2J_MJJ_0_60':'darkgoldenrod', 
'QQ2HQQ_GE2J_MJJ_60_120':'goldenrod', 'QQ2HQQ_GE2J_MJJ_120_350':'gold','QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200':'lemonchiffon',
'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25':'khaki', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25':'olive',
'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25':'olivedrab', 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25':'yellowgreen'}

color_map = {'ggH':'blue', 'qqH':'orange', 'WH':'green','ZH':'springgreen','ttH':'purple','tH':'yellow'}  

#define object
folder = 'multi6'
#six class
mc_object = multiClass(dataframe_ggh, dataframe_vbf, dataframe_vh, dataframe_tth, dataframe_thw, dataframe_thq, features, 
folder, class_list, [], color_map)
#stxs
#mc_object = multiClass(dataframe_ggh, dataframe_vbf, dataframe_vh, dataframe_tth, dataframe_thw, dataframe_thq, features, 
#folder, class_list2, stxs_list, color_map2)

#pre-processing
mc_object.combine_dataframe()
mc_object.feature_engineered()
mc_object.equalise_weight(equalise=True)

#mc_object.take_log(Gev_list) #FIXME: don't take log for now
mc_object.missing_values(-10)

#train classifier
#['fitfthclass', 'efeatures']
mc_object.train_classifier(layers=2, epoch=100, dropout=False, batch_size=64, neurons=100, reg_size=0.001, reg_type='l1', 
    learning_rate=0.00010, sch_decay = 0.05, patience = 5, weight_scaler=10**5, comp_name=None, extra_label='sixclass',)
#mc_object.train_bdt(bdtname='no_eq')

#plotting stuff
mc_object.plot_output(argmax=True)
mc_object.plot_roc_curves()
mc_object.plot_confusion_matrix()

