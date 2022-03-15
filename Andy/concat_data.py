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

dataframe_ggh = pd.read_csv('%s/ggH_multiclass_ann_df_2017.csv'%options.data)
dataframe_vbf = pd.read_csv('%s/VBF_multiclass_ann_df_2017.csv'%options.data)
dataframe_vh = pd.read_csv('%s/VH_multiclass_ann_df_2017.csv'%options.data)
dataframe_tth = pd.read_csv('%s/ttH_multiclass_ann_df_2017.csv'%options.data)
dataframe_thw = pd.read_csv('%s/tHW_multiclass_ann_df_2017.csv'%options.data)
dataframe_thq = pd.read_csv('%s/tHq_multiclass_ann_df_2017.csv'%options.data)

df_Diphoton40To80 = pd.read_csv('%s/Diphoton40To80_multiclass_ann_df_2017.csv'%options.data)
df_Diphoton80ToInf = pd.read_csv('%s/Diphoton80ToInf_multiclass_ann_df_2017.csv'%options.data)
df_GJet20To40 = pd.read_csv('%s/GJet20To40_multiclass_ann_df_2017.csv'%options.data)
df_GJet20ToInf = pd.read_csv('%s/GJet20ToInf_multiclass_ann_df_2017.csv'%options.data)
df_GJet40ToInf = pd.read_csv('%s/GJet40ToInf_multiclass_ann_df_2017.csv'%options.data)
df_QCD_Pt30to40 = pd.read_csv('%s/QCD_Pt-30to40_multiclass_ann_df_2017.csv'%options.data)
df_QCD_Pt30toInf = pd.read_csv('%s/QCD_Pt-30toInf_multiclass_ann_df_2017.csv'%options.data)
df_QCD_Pt40toInf = pd.read_csv('%s/QCD_Pt-40toInf_multiclass_ann_df_2017.csv'%options.data)


fold = 10
for df in [dataframe_ggh, dataframe_vbf, dataframe_vh, df_Diphoton40To80, df_Diphoton80ToInf, df_GJet20To40, df_GJet20ToInf, df_GJet40ToInf, df_QCD_Pt30to40, df_QCD_Pt40toInf]:
    print(len(df))
    df = df.sample(len(df)/fold)
    print(len(df))
    df['weight'] = df['weight']*fold

#dataframe_full = pd.concat([dataframe_ggh,dataframe_vbf,dataframe_vh,dataframe_tth,dataframe_thw,dataframe_thq,df_Diphoton40To80,df_Diphoton80ToInf,df_GJet20To40,df_GJet20ToInf,df_GJet40ToInf,df_QCD_Pt30to40,df_QCD_Pt30toInf,df_QCD_Pt40toInf])
dataframe_full = pd.concat([dataframe_ggh,dataframe_vbf,dataframe_vh,df_Diphoton40To80,df_Diphoton80ToInf,df_GJet20To40,df_GJet20ToInf,df_GJet40ToInf,df_QCD_Pt30to40,df_QCD_Pt30toInf,df_QCD_Pt40toInf])

def change_df_dtype(df):
    print(df.info())
    cols = df.select_dtypes(include=[np.float64]).columns
    df[cols] = df[cols].astype(np.float32)
    cols = df.select_dtypes(include=[np.int64]).columns
    df[cols] = df[cols].astype(np.int32)
    print(df.info())
    return df

dataframe_full = change_df_dtype(dataframe_full)
#dataframe_full.to_csv('%s/data_full_prod.csv'%(options.data), index=False)
dataframe_full.to_csv('%s/data_full_stxs.csv'%(options.data), index=False)