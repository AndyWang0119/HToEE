import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("cms10_6_HP")
import gc
from MultiClass import multiClass
import os

#loading data
parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('Required Arguments')
required_args.add_argument('-d','--data', action='store', help='Input data dir', required=True)
options=parser.parse_args()

try:
    dataframe_full = pd.read_csv('Andy/rec_data/datafull_stxs_bkg.csv')
    loadcomplieddata = 0
    print('loaded reconstructed data')

except IOError:
    dataframe_full = pd.read_csv('%s/data_full.csv'%options.data)
    loadcomplieddata = 1

    print('Loaded all data')


features = ['diphotonPt','diphotonCosPhi', 'leadPhotonPtOvM', 'subleadPhotonPtOvM', 'leadPhotonEta', 'subleadPhotonEta',
    'dijetAbsDEta', 'dijetDPhi','dijetMinDRJetPho', 'dijetMass','dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'dijetCentrality',
    'leadJetDiphoDPhi', 'subleadJetDiphoDPhi', 'leadJetDiphoDEta', 'subleadJetDiphoDEta', 
    'leadPhotonIDMVA', 'subleadPhotonIDMVA', 'subsubleadPhotonIDMVA',
    #'subsubleadJetDiphoDPhi', 'subsubleadJetDiphoDEta',
    'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL','subleadJetEn', 'subleadJetPt', 'subleadJetEta', 'subleadJetPhi','subleadJetQGL',
    'subsubleadJetEn', 'subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetPhi','subsubleadJetQGL',

   'leadJetBTagScore','subleadJetBTagScore', 'subsubleadJetBTagScore',
    #'leadJetPUJID', 'subleadJetPUJID', 'subsubleadJetPUJID', 
    'nSoftJets', 'metPt', 'metPhi', 'metSumET', 'metSignificance',
    'leadElectronMass', 'leadMuonMass', 'subleadElectronMass', 'subleadMuonMass',
    #'subsubleadElectronMass', 'subsubleadMuonMass',
    'leadElectronEn', 'leadMuonEn', 'leadElectronPt', 'leadMuonPt', 'leadElectronEta', 'leadMuonEta', 'leadElectronPhi', 'leadMuonPhi',
    'leadElectronCharge', 'leadElectronConvVeto', 'leadMuonCharge',
    #quite a lot of sublead lepton are missing
    'subleadElectronEn', 'subleadMuonEn', 
    'subleadElectronPt', 'subleadMuonPt', 'subleadElectronEta', 'subleadMuonEta', 'subleadElectronPhi', 'subleadMuonPhi',
    'subleadElectronCharge', 'subleadElectronConvVeto', 'subleadMuonCharge',
    #'subsubleadElectronEn', 'subsubleadElectronPt', 
    #'subsubleadMuonEn',  'subsubleadMuonPt', 'subsubleadElectronEta', 'subsubleadMuonEta', 'subsubleadElectronPhi', 'subsubleadMuonPhi',
    #'subsubleadElectronCharge', 'subsubleadElectronConvVeto', 'subsubleadMuonCharge',
    ]


Gev_list = ['diphotonPt', 'leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt','metPt', 'subsubleadJetEn', 'subsubleadJetPt',
'leadElectronEn', 'leadMuonEn', 'leadElectronPt', 'leadMuonPt', 'subsubleadJetEn', 'subsubleadJetPt',
'leadElectronEn', 'leadMuonEn', 'leadElectronPt', 'leadMuonPt', 'subleadElectronEn', 'subleadMuonEn', 'subleadElectronPt', 'subleadMuonPt',
'dijetMass','ptHjj']

class_list2 = ['QQ2HQQ_0J1J','QQ2HQQ_GE2J_MJJ_60_120', 
'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25', 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']

stxs_list = ['QQ2HQQ_FWDH', 'QQ2HQQ_0J', 'QQ2HQQ_1J', 'QQ2HQQ_GE2J_MJJ_0_60', 'QQ2HQQ_GE2J_MJJ_60_120', 'QQ2HQQ_GE2J_MJJ_120_350',
'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25', 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']

color_map3 = {'QQ2HQQ_0J1J':'blue',
'QQ2HQQ_GE2J_MJJ_60_120':'orange', 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200':'green',
'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25':'purple', 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25':'black',
'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25':'darkred', 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25':'peru'}


#define object
folder = 'multi10'
#stxs
mc_object = multiClass(dataframe_full, features, folder, class_list2, stxs_list, color_map3, data_label='stxs_bkg', recon=loadcomplieddata)

#save memory
del dataframe_full
gc.collect()

#pre-processing
mc_object.combine_dataframe(prod=False)
mc_object.feature_engineered(lep_num = True, PUJID = True)
mc_object.equalise_weight(equalise=False)

mc_object.take_log(Gev_list)
mc_object.missing_values(-10, scaler=False)

#train classifier
mc_object.train_classifier(layers=2, epoch=100, dropout=True, batch_size=64, neurons=50, reg_size=0.0001, reg_type='l2', 
    learning_rate=0.0001, sch_decay = 0.01, sch_period=1, patience = 10, weight_scaler=10**5, 
    comp_name=None, extra_label='bkg', drop_size=0.2)

#mc_object.stxs_cut()

#plotting stuff
#mc_object.plot_output()
#mc_object.plot_roc_curves()
mc_object.plot_confusion_matrix(y_pred=mc_object.y_pred_test)
#mc_object.plot_confusion_matrix(y_pred=mc_object.y_pred_test_cut,label='cut')
mc_object.output_class_stack()

