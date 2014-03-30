import numpy
from trained_model import *

class TMOneSentence(TrainedModel):
    def __init__(self, pklprefix, seed=0, learnrate = 0.0125, reg = 0.00005, xsamples=400, noise="False", sentence_num = 0 ):
        self.string_desc_base = "one_sentence-%d"%sentence_num
        self.sentence_num = sentence_num
        TrainedModel.__init__( self, seed=seed, pklprefix=pklprefix, learnrate=learnrate, reg=reg, xsamples = xsamples, noise=noise )
        
        self.MonitorServer = MonitorServerOneSentence
    
    def predict_each_original_sample( self, dataset):
        #dataset = self.parse_yaml().dataset
        
        bestMLP = self.load_bestMLP()
        X = bestMLP.get_input_space().make_batch_theano()
        Y = bestMLP.fprop(X)
        pred_next_sample = theano.function( [X[0]], Y )
        
        sentence_examples = dataset.get(['features'], range(dataset.num_examples))[0]
        sentence_targets  = dataset.get(['targets'],  range(dataset.num_examples))[0]
        
        preds = pred_next_sample( sentence_examples )
        preds = preds.reshape( (1,preds.shape[0]) )
        sentence_targets = sentence_targets.reshape( (1,preds.shape[1]) )
        
        ret = numpy.vstack( (sentence_targets,preds) )
        ret = numpy.hstack( ( numpy.zeros( (2, len(dataset.raw_wav[0])-preds.shape[1]) ), ret ) )
        return ret
    
#    def generate_pcm( self, sigmacoeffs = [0.1], init_indices=[0] ):
#        wave, raw_wav, dataset = TrainedModel.generate_pcm( self, sigmacoeffs, init_indices, None )
#        original = (dataset.raw_wav[0].astype('float')-dataset._mean)/dataset._std
#        preds = self.predict_each_original_sample()
#        return numpy.vstack( (wave, original, preds) )
    
    def generate_pcm_with_orig_and_pred( self, sigmacoeffs = [0.1], init_indices=[0] ):
        wave, raw_wav, dataset = self.generate_pcm( sigmacoeffs, init_indices, None )
        original = (dataset.raw_wav[0].astype('float')-dataset._mean)/dataset._std
        preds = self.predict_each_original_sample(dataset)
        return numpy.vstack( (wave, original, preds) )
    
    def generate_with_restarts( self, length ):
        dataset = self.parse_yaml().dataset
        init_idcs = range( length, len(dataset.raw_wav[0])-length, length ) 
        wave, _, _ = self.generate_pcm( [0], init_idcs, length )
        wave = wave.reshape( (1,length*wave.shape[0]) )
        original = (dataset.raw_wav[0].astype('float')-dataset._mean)/dataset._std
        
        return numpy.vstack( (wave, original[init_idcs[0]:init_idcs[-1]+length]) )

    def mse_with_restarts( self, length ):
        wav = self.generate_with_restarts( self, length )
        return sum( (wav[0,:]-wav[1,:])**2 )

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
                noise: """ + self.noise + """
            }"""
    
    def monitoringdatasetyaml( self ):
        return """'train': *train"""
    
    def monitoringextensionyaml( self ):
        return """!obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                    channel_name: 'train_objective',
                    save_path: \"""" + self.bestMLPpath + """\",
                }"""

class MonitorServerOneSentence(MonitorServer):
    def do_generatepcmwithpreds( self, args ):
        if 'sigmas' in args.keys():
            sigmas = map( lambda x: float(x), args['sigmas'][0].split(',') )
        else:
            sigmas = [0]
        if 'init_idx' in args.keys():
            init_idcs = [ int(args['init_idx'][0]) ]
        else:
            init_idcs = [0]
        try:
             arr = self.tm.generate_pcm_with_orig_and_pred( sigmas, init_idcs )
        except:
             self.send_python_error()
        else:
             self.send_ascii_encoded_array( arr )

    def do_generatewithrestarts( self, args ):
        if 'length' in args.keys():
            length = int( args['length'][0] )
        else:
            length = 3000
        try:
             arr = self.tm.generate_with_restarts( length )
        except:
             self.send_python_error()
        else:
            self.send_ascii_encoded_array( arr )
    
    def do_restartmse( self, args ):
        if 'length' in args.keys():
            length = int( args['length'][0] )
        else:
            length = 3000
        try:
             mse = self.tm.mse_with_restarts( length )
        except:
             self.send_python_error()
        else:
            self.wfile.write( str(mse) ) 

if __name__=="__main__":
    print "Arguments: whatdo pklprefix learnrate reg noise sentence_num"
    whatdo = sys.argv[1]
    pklprefix = sys.argv[2]
    learnrate = float(sys.argv[3])
    reg = float(sys.argv[4])
    noise = sys.argv[5]
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
