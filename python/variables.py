#electron vars
nominal_vars = ['diphotonMass', 'leadPhotonElectronVeto', 'subleadPhotonElectronVeto',
 		        'weight', 'leadPhotonIDMVA', 'subleadPhotonIDMVA', 'subsubleadPhotonIDMVA',

                 'diphotonSigmaMoM', 'leadPhotonPhi', 'subleadPhotonPhi',

                'diphotonPt','diphotonCosPhi',
                'leadPhotonPtOvM', 'subleadPhotonPtOvM',
                'leadPhotonEta', 'subleadPhotonEta',
                'dijetAbsDEta', 'dijetDPhi','dijetMinDRJetPho', 'dijetMass', 
                'dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 'dijetCentrality',
                'leadJetDiphoDPhi', 'subleadJetDiphoDPhi', 'leadJetDiphoDEta', 'subleadJetDiphoDEta',
                'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', 
                'subleadJetEn', 'subleadJetPt', 'subleadJetEta', 'subleadJetPhi','subleadJetQGL',
               ]

#for MVA training, hence not including masses
gev_vars     = ['leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt', 'subsubleadJetEn', 'subsubleadJetPt', 
                'leadElectronEn', 'leadElectronPt', 'subleadElectronEn', 'subleadElectronPt',
                'leadElectronPToM', 'subleadElectronPToM', 'dijetMass', 'dielectronPt'
               ]

gen_vars     = ['genWeight'] 

