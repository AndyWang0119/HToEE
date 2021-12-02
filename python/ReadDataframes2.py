import argparse
import pandas as pd
import glob
import numpy as np
import scipy as sp
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import plot_importance
import pickle

parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('Required Arguments')
required_args.add_argument('-b','--bdt', action='store', help='Input bdt dataframe dir', required=True)
options=parser.parse_args()

dataframe = pd.read_csv('%s/data.csv'%options.bdt)

y_train = dataframe[dataframe['type']=='train']['y']
y_pred_train = dataframe[dataframe['type']=='train']['y_pred']
train_weights = dataframe[dataframe['type']=='train']['weights']

y_test = dataframe[dataframe['type']=='test']['y']
y_pred_test = dataframe[dataframe['type']=='test']['y_pred']
test_weights = dataframe[dataframe['type']=='test']['weights']

import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
plt.style.use("cms10_6_HP")

def plot_roc(y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = False):
    label = ''
    color = 'red'
    if abs:
        train_weights = np.abs(train_weights)
        test_weights = np.abs(test_weights)
        label = 'absolute'
        color = 'blue'
    
    
    bkg_eff_train, sig_eff_train, _ = roc_curve(y_train, y_pred_train, sample_weight=train_weights)
    bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test, sample_weight=test_weights)

    fig = plt.figure(1)
    axes = fig.gca()
    axes.plot(bkg_eff_train, sig_eff_train, color=color, label='Train %s'%label)
    axes.plot(bkg_eff_test, sig_eff_test, color=color, label='Test %s'%label)
    axes.set_xlabel('Background efficiency', ha='right', x=1, size=13)
    axes.set_xlim((0,1))
    axes.set_ylabel('Signal efficiency', ha='right', y=1, size=13)
    axes.set_ylim((0,1))
    axes.legend(bbox_to_anchor=(0.97,0.97))
    axes.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)

    plt.savefig('plotting/plots/ggH_BDT/ROC_%s'%label)

roc = 0
if roc:
    plot_roc(y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = False) 
    plot_roc(y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, abs = True) 


clf = pickle.load(open('models/ggH_BDT_clf.pickle.dat'))
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
clf.get_booster().feature_names = features

for type in ['gain', 'weight', 'cover']:
    ax = plot_importance(clf, importance_type=type, show_values=True, title='Importance by %s'%type) 

    fig = ax.figure
    plt.savefig('plotting/plots/ggH_BDT/feature_importance_%s.png'%type)
