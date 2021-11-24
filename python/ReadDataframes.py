import argparse
import pandas as pd
import glob
import numpy as np
import scipy as sp

parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('Required Arguments')
required_args.add_argument('-m','--mc', action='store', help='Input MC dataframe dir', required=True)
required_args.add_argument('-d','--data', action='store', help='Input data dataframe dir', required=True)
options=parser.parse_args()

files_mc_csv = glob.glob("%s/*.csv"%options.mc)
files_mc_data = glob.glob("%s/*.csv"%options.data)

dataframes = []
for f in files_mc_csv:
  dataframes.append( pd.read_csv(f) )
  print " --> Read: %s"%f
for f in files_mc_data:
  dataframes.append( pd.read_csv(f) )
  print " --> Read: %s"%f

df = pd.concat( dataframes, sort=False, axis=0 )
print " --> Successfully read dataframes. Printing first five events"
print df.head()

#plotting the different signal in a signle plots for a selection of variables
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
plt.style.use("cms10_6_HP")

#FIXME: temporarily added to this file, since the .csv does not have the feature now
df['diphotonDR'] = np.sqrt((df['leadPhotonEta'] - df['subleadPhotonEta'])**2 + (np.arccos(df['diphotonCosPhi']))**2)

variables_to_plot =  ['diphotonPt','diphotonCosPhi',      
'leadPhotonPtOvM', 'subleadPhotonPtOvM',
'leadPhotonEta', 'subleadPhotonEta',
'dijetAbsDEta', 'dijetDPhi','dijetMinDRJetPho', 'dijetMass', 
'dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'dijetCentrality',
'leadJetDiphoDPhi', 'subleadJetDiphoDPhi', 'leadJetDiphoDEta', 'subleadJetDiphoDEta',
'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', 
'subleadJetEn', 'subleadJetPt', 'subleadJetEta', 'subleadJetPhi','subleadJetQGL',
'diphotonDR'
]


import yaml
with open('plotting/var_to_xrange.yaml', 'r') as plot_config_file:
  plot_config        = yaml.load(plot_config_file)
  var_to_xrange = plot_config['var_to_xrange']

#1d hist
hist=0
if hist:
  n_bins = 40

  #variables_to_plot = ['diphotonPt']
  signal_to_plot = ['ggH', 'VBF', 'VH', 'ttH']
  colours = ['r','g','b','y']

  for var in variables_to_plot:
    fig  = plt.figure(1)
    axes = fig.gca()
    bins = np.linspace(var_to_xrange[var][0], var_to_xrange[var][1], n_bins)
    for i,sig in enumerate(signal_to_plot):
      signal = df[df['proc']==sig] #plot the perticular signal process
      signal = signal[signal[var]>var_to_xrange[var][0]]
      signal = signal[signal[var]<var_to_xrange[var][1]] #remove events outside plotting range
      norm = signal['weight'].agg('sum') #normalization
      axes.hist(signal[var], bins=bins, label=sig+r' ($\mathrm{H}\rightarrow\gamma\gamma$) ', weights=signal['weight']/norm, histtype='step', color=colours[i], zorder=10)
    axes.legend(ncol=2, prop={'size':9})
    axes.set_xlabel('%s' %var)
    axes.set_ylabel('A Normalised Unit')
    #axes.text(0, 1.01, r'\textbf{CMS} %s'%label, ha='left', va='bottom', transform=axes.transAxes, size=14)
    plt.savefig('plotting/plots/all_sig/%s.png' %var)
    plt.close()


roc = 0
if roc:
  dist = 'dijetCentrality'
  ratio1 = []
  ratio2 = []
  for cut in np.arange(var_to_xrange[dist][0],var_to_xrange[dist][1], 0.1):
    print(cut)
    bkg_proc = ['VH','ttH','ggH']
    sig_proc = 'VBF'
    correct = 0
    incorrect = 0
    total1 = 0
    total2 = 0
    signal = df[df[dist]!= -999.0] #do not count the Nan value
    for proc in bkg_proc:
      correct += signal[signal['proc']==proc].query('%s < %s' %(dist,cut))['weight'].agg('sum')
      total1 +=  signal[signal['proc']==proc]['weight'].agg('sum')
    incorrect += signal[signal['proc']==sig_proc].query('%s < %s' %(dist,cut))['weight'].agg('sum')
    total2 += signal[signal['proc']==sig_proc]['weight'].agg('sum')
    ratio1.append(correct/total1)
    ratio2.append(incorrect/total2)
  fig  = plt.figure(1)
  axes = fig.gca()
  print(ratio1, ratio2)
  plt.plot(ratio1,ratio2)
  axes.set_xlabel('Bakcground Removed')
  axes.set_ylabel('Signal Removed')
  axes.set_title('ROC curve of VBF by %s' %dist)
  plt.savefig('plotting/plots/ROC.png')
    

#plot the 2D correlation, both scatter plots and 2d hist in subplots
scatter_plot=0
corr_var_pair = [['leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL'],
['diphotonPt','diphotonCosPhi','leadPhotonPtOvM', 'subleadPhotonPtOvM'],
['dijetAbsDEta', 'dijetDPhi','dijetMinDRJetPho', 'dijetMass','dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'dijetCentrality',],
['diphotonDR','diphotonPt']]
#corr_variables =  ['diphotonPt','diphotonCosPhi','leadPhotonPtOvM', 'subleadPhotonPtOvM']
#corr_variables = ['dijetAbsDEta', 'dijetDPhi','dijetMinDRJetPho', 'dijetMass','dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'dijetCentrality',]
#corr_variables = ['leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL']
#corr_var_pair = [['diphotonDR','diphotonPt']]


if scatter_plot:
  cmap_list = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples]
  colours = ['blue','orange','green','purple']
  for corr_variables in corr_var_pair:
    for i, var1 in enumerate(corr_variables):
      for j, var2 in enumerate(corr_variables):
        if i < j:
          fig, ax = plt.subplots(figsize = (14,8), nrows=2, ncols=2, sharex=True, sharey=True)
          fig2, ax2 = plt.subplots(figsize = (13,8), nrows=2, ncols=2, sharex=True, sharey=True)
          
          for index,sig in enumerate(['ggH', 'VBF', 'VH', 'ttH']):

              signal = df[df['proc']==sig]

              #2d hist
              bins = (50,50)
              range = np.array([[var_to_xrange[var1][0],var_to_xrange[var1][1]],[var_to_xrange[var2][0],var_to_xrange[var2][1]]])
              h = ax2[index%2,index//2].hist2d(signal[var1],signal[var2],bins=bins,range=range, cmap=cmap_list[index])
              fig2.colorbar(h[3], ax=ax2[index%2,index//2])

              #scatter plot
              #ax[index%2,index//2].scatter(signal[var1], signal[var2], s=3, alpha=0.005, label=sig, c=colours[index])

              ax[index%2,index//2].set_xlim(var_to_xrange[var1][0], var_to_xrange[var1][1])
              ax[index%2,index//2].set_ylim(var_to_xrange[var2][0], var_to_xrange[var2][1])
              ax[index%2,index//2].set_xlabel(var1)
              ax[index%2,index//2].set_ylabel(var2)
              ax[index%2,index//2].legend(prop={'size':9})
              ax[index%2,index//2].set_title(sig)

              ax2[index%2,index//2].set_xlim(var_to_xrange[var1][0], var_to_xrange[var1][1])
              ax2[index%2,index//2].set_ylim(var_to_xrange[var2][0], var_to_xrange[var2][1])
              ax2[index%2,index//2].set_xlabel(var1)
              ax2[index%2,index//2].set_ylabel(var2)
              ax2[index%2,index//2].set_title(sig)

          #fig.savefig('plotting/plots/scatter_plot/%s_against_%s.png' %(var1,var2))
          fig2.savefig('plotting/plots/2dhist/%s_against_%s.png' %(var1,var2))
          plt.close()

# Compute the correlation matrix of all training features
correlation = 1
if correlation:
  import seaborn as sns
  sig = 'ggH' #take ggH as an example

  signal = df[df['proc']==sig]
  signal = signal[variables_to_plot]

  #remove -999.0 rows
  for var in signal.columns:
    signal = signal[signal[var] != -999.0]
    print(len(signal))
  
  corr_from_scratch = 0
  if corr_from_scratch:
    corr = np.zeros([len(signal.columns),len(signal.columns)])
    for i,x in enumerate(signal.columns):
      for j,y in enumerate(signal.columns):
        if j >= i: #reduce the workload
          xmean = signal[x].agg('mean')
          ymean = signal[y].agg('mean')
          n = len(signal[x])

          sum1 = np.sum((signal[x]-xmean)*(signal[y]-ymean))
          sum2 = np.sum((signal[x]-xmean)**2)
          sum3 = np.sum((signal[y]-ymean)**2)
          #for i in range(len(signal[x])):
          #  sum1 += (signal[x][i]-xmean)*(signal[y][i]-ymean)
          #  sum2 += (signal[x][i]-xmean)**2
          #  sum3 += (signal[y][i]-ymean)**2
          r = sum1/(np.sqrt(sum2)*np.sqrt(sum3))
          corr[i,j] = r
          corr[j,i] = r #symmetric matrix
    corr = pd.DataFrame(corr, index=signal.columns, columns=signal.columns)
  else:
    corr = signal.corr()

  print(corr)
  
  f, ax = plt.subplots(figsize=(11, 9))
  cmap = sns.diverging_palette(230, 20, as_cmap=True)
  corr = corr.round(2)
  sns.heatmap(corr, cmap=cmap, vmax=1.0, center=0, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws={"size": 9})
  plt.savefig('plotting/plots/scatter_plot/correlation.png')
  

