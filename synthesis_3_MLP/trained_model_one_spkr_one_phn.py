import trained_model

class TMOneSpkrOnePhn(TrainedModel):
    def __init__(self, pklprefix, seed=0, learnrate = 0.0125, reg = 0.00005, xsamples=400, noise="False" ):
        self.speaker_id = speaker_id
        self.phone = phone
        self.string_desc_base = "one_spkr_one_phn-%d-%d"%(speaker_id,phone)
        super(self).__init( seed=seed, pklprefix=pklprefix, learnrate=learnrate, reg=reg, xsamples = xsamples, noise=noise )

    def datasetyaml( self, trainorvalid ):
        if trainorvalid=='train':
            s = "train_train"
            noiserow = """noise: """ + str(self.noise) + ""","""
        elif trainorvalid=='valid':
            s = "train_valid"
            noiserow = ""
        else:
            raise str(trainorvalid) + " is not a valid choice for train set"
        return """!obj:research.code.pylearn2.datasets.timit.TIMITOnTheFly {
                which_set: """ + s + """,
                frame_length: 1,
                frames_per_example: """ + str(self.xsamples )+ """,
                """ + noiserow + """
                speaker_filter: [""" + str(self.speaker_id) + """],
                phone_filter: [""" + str(self.phone) + """],
                mid_third: True
            }"""
    
    def monitoringdatasetyaml( self ):
        return """'train': *train,
                   'valid': """ + self.datasetyaml('valid')
    
    def monitoringextensionyaml( self ):
        return """!obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                    channel_name: 'valid_objective',
                    save_path: \"""" + self.bestMLPpath + """\",
                }"""


if __name__=="__main__":
    print "Arguments: pklprefix learnrate reg speaker_id phone xsamples noise seed"
    whatdo = sys.argv[1]
    pklprefix = sys.argv[2]
    learnrate = float(sys.argv[3])
    reg = float(sys.argv[4])
    speaker_id = int(sys.argv[5])
    phone = int(sys.argv[6])
    if sys.argv[7]=="False":
      noise = False
    else:
      noise = float(sys.argv[7])
    global tm
    tm  = TrainedModel(pklprefix, learnrate = learnrate, reg=reg, speaker_id=speaker_id, phone=phone,noise=noise)
    if whatdo=='train' or whatdo=='forcetrain':
        if os.path.isfile(tm.progressMLPpath) and whatdo=='train':
            print "Are you sure you want to train?",tm.progressMLPpath, "exists. If so use forcetrain option."
        else:
            tm.train()
    elif whatdo=='server':
        tm.start_web_server( True )
    elif whatdo=='monitor':
        tm.training_graph()
    elif whatdo=='showmse':
        mses = tm.mses()
        print "train", mses[0], "valid",mses[1]
    elif whatdo=='generate':
        tm.generate()
    elif whatdo=='yaml':
        print tm.yaml()

