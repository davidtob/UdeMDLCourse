import numpy
from trained_model import *

class TMOneSentence(TrainedModel):
    def __init__(self, pklprefix, seed=0, learnrate = 0.0125, reg = 0.00005, xsamples=400, ysamples=1, noise="False", sentence_num = 0, noise_decay = False ):
        self.string_desc_base = "one_sentence-s-%d"%sentence_num
        self.sentence_num = sentence_num
        TrainedModel.__init__( self, seed=seed, pklprefix=pklprefix, learnrate=learnrate, reg=reg, xsamples = xsamples, ysamples = ysamples, noise=noise, noise_decay = noise_decay )
        
        self.MonitorServer = MonitorServerOneSentence
    
    def predict_each_original_sample( self, dataset ):
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
        dataset = self.dataset_for_generation()
        init_idcs = range( length, len(dataset.raw_wav[0])-length, length ) 
        wave, _, _ = self.generate_pcm( [0], init_idcs, length )
        wave = wave.reshape( (1,length*wave.shape[0]) )
        original = (dataset.raw_wav[0].astype('float')-dataset._mean)/dataset._std
        
        return numpy.vstack( (wave, original[init_idcs[0]:init_idcs[-1]+length]) )

    def mse_with_restarts( self, length ):
        wav = self.generate_with_restarts( length )
        return sum( (wav[0,:]-wav[1,:])**2 )/wav.shape[1]

    def recursion_errors( self, length=1024, num_idcs = None ):
        dataset = self.dataset_for_generation()
        original = (dataset.raw_wav[0].astype('float')-dataset._mean)/dataset._std
        if num_idcs == None:
            init_idcs = range(len(dataset.raw_wav[0]) - length)
        else:
            init_idcs = numpy.random.randint( 0, len(dataset.raw_wav[0]) - length, num_idcs )
        print "generating"
        wave, _, _ = self.generate_pcm( [0], init_idcs, length )
        errs = numpy.zeros( length )
        apa = 0
        errs = numpy.zeros( (len(init_idcs), length) )
        for i in range(wave.shape[0]):
            #print (wave[i,:] - original[i:i+length])**2
            errs[i,:] = wave[i,:] - original[init_idcs[i]:init_idcs[i]+length]
            #errs[i,:] = wave[i,:] - original[i:i+length]
            #print wave[i,self.xsamples]#,original[i+self.xsamples]
        return errs

    def recursion_rmse( self, length = 1024, num_idcs = None ):
        #dataset = self.dataset_for_generation()
        #original = (dataset.raw_wav[0].astype('float')-dataset._mean)/dataset._std
        #length = 1024
        #init_idcs = range( len(dataset.raw_wav[0]) - length )
        #print "generating"
        #wave, _, _ = self.generate_pcm( [0], init_idcs, length )
        #errs = numpy.zeros( length )
        #apa = 0
        #for i in range(wave.shape[0]):
        #    #print (wave[i,:] - original[i:i+length])**2
        #    errs += (wave[i,:] - original[i:i+length])**2
        #    apa += (wave[i,self.xsamples] - original[i+self.xsamples])**2
        #    #print wave[i,self.xsamples]#,original[i+self.xsamples]
        #errs = numpy.sqrt( errs/wave.shape[0] )
        errs = self.recursion_errors( length = length, num_idcs = num_idcs )
        return numpy.sqrt( numpy.sum(errs**2/errs.shape[0],0) )

    def datasetyaml( self, trainorvalid, withnoise = True ):
        #if trainorvalid!='train':
        #    raise str(trainorvalid) + " is not a valid choice for train set"
        ret= """!obj:research.code.pylearn2.datasets.timit.TIMITOnTheFly {
                which_set: 'train',
                frame_length: 1,
                frames_per_example: """ + str(self.xsamples )+ """,
                output_frames_per_example: """ + str(self.ysamples) + """,
                start: """ + str(self.sentence_num) + """,
                stop: """ + str(self.sentence_num+1) + """,
                audio_only: True"""
        if withnoise:
            ret = ret + """,
                            noise: """ + str(self.noise) + """,
                            noise_decay: """ + str(self.noise_decay) + """,
                         }"""
        else:
            ret = ret + """}"""
        return ret
    
    def monitoringdatasetyaml( self ):
        return """'train': *train,
                  'valid': """ + self.datasetyaml( 'train', False )
    
    def monitoringextensionyaml( self ):
        return """!obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                    channel_name: 'valid_objective',
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
    print "Arguments: whatdo pklprefix learnrate sentence_num xsamples ysamples reg noise noise_decay"
    whatdo = sys.argv[1]
    pklprefix = sys.argv[2]
    learnrate = float(sys.argv[3])
    sentence_num = int(sys.argv[4])
    xsamples = int(sys.argv[5])
    ysamples = int(sys.argv[6])
    reg = float(sys.argv[7])
    noise = sys.argv[8]
    if len(sys.argv )>=10:
        if sys.argv[9]=="True":
            noise_decay = True
        else:
            noise_decay = False
    else:
        noise_decay = False
        
    print noise_decay
    
    global tm
    tm  = TMOneSentence(pklprefix, learnrate = learnrate, reg=reg,noise=noise, sentence_num=sentence_num, noise_decay=noise_decay, xsamples=xsamples,ysamples=ysamples )
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
    elif whatdo=='restartmse':
        print tm.mse_with_restarts( 3000 )
    elif whatdo=='generate':
        tm.generate()
    elif whatdo=='yaml':
        print tm.yaml()
    elif whatdo=='recursionrmse':
        print tm.recursion_rmse(length=410,num_idcs=None).tolist()
