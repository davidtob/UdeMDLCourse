import numpy
from trained_model import *

class TMOneSentence(TrainedModel):
    def __init__(self, pklprefix, seed=0, learnrate = 0.0125, reg = 0.00005, xsamples=400, noise=False, sentence_num = 0 ):
        self.string_desc_base = "one_sentence-%d"%sentence_num
        self.sentence_num = sentence_num
        TrainedModel.__init__( self, seed=seed, pklprefix=pklprefix, learnrate=learnrate, reg=reg, xsamples = xsamples, noise=noise )
    
    def generate_pcm( self, sigmacoeffs = [0.1], init_indices=[0] ):
        raw_wav, dataset = TrainedModel.generate_pcm( self, sigmacoeffs, init_indices, None )
        original = dataset.raw_wav[0]
        return numpy.vstack( (raw_wav, original) )

    def datasetyaml( self, trainorvalid ):
        if trainorvalid!='train':
            raise str(trainorvalid) + " is not a valid choice for train set"
        return """!obj:research.code.pylearn2.datasets.timit.TIMITOnTheFly {
                which_set: 'train',
                frame_length: 1,
                frames_per_example: """ + str(self.xsamples )+ """,
                start: """ + str(sentence_num) + """,
                stop: """ + str(sentence_num+1) + """,
                audio_only: True,
            }"""
    
    def monitoringdatasetyaml( self ):
        return """'train': *train"""
    
    def monitoringextensionyaml( self ):
        return """!obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                    channel_name: 'train_objective',
                    save_path: \"""" + self.bestMLPpath + """\",
                }"""

if __name__=="__main__":
    print "Arguments: whatdo pklprefix learnrate reg noise sentence_num"
    whatdo = sys.argv[1]
    pklprefix = sys.argv[2]
    learnrate = float(sys.argv[3])
    reg = float(sys.argv[4])
    if sys.argv[5]=="False":
      noise = False
    else:
      noise = float(sys.argv[5])
    sentence_num = int(sys.argv[6])
    global tm
    tm  = TMOneSentence(pklprefix, learnrate = learnrate, reg=reg,noise=noise, sentence_num=sentence_num )
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