import numpy
from string import Template
from pylearn2.config import yaml_parse
import pylab as plt
import sys
import cPickle
import theano
import os
import scipy.io.wavfile
import itertools

class TrainedModel:
    # This class should
    # 1. Be able to randomize hyper parameters
    # 2. store hyperparameters
    # 3. Produce yaml file
    # 4. Train based on yaml file
    # 5. Monitor training progress
    # 6. Compute MSE of trained model
    # 7. Generate from trained model
    def __init__(self, pklprefix, learnrate = 0.0125, reg = 0.00005, speaker_id=104, phone=28, xsamples=400, noise=0.1, seed=0 ):
        self.pklprefix = pklprefix
        self.learnrate = learnrate
        self.speaker_id = speaker_id
        self.phone = phone
        self.xsamples = xsamples
        self.noise = noise
        self.reg = reg
        self.seed = seed
        numpy.random.seed( self.seed )

        self.string_desc = "one-speaker-one-phone-noise-wide-%f-%d-%d-%d-%f-%d"%(learnrate,speaker_id,phone,xsamples,noise,seed)
        self.progressMLPpath = self.pklprefix + "/progress-" + self.string_desc + ".pkl"
        self.bestMLPpath     = self.pklprefix + "/best-" + self.string_desc + ".pkl"
    
    def load_progressMLP( self ):
        return cPickle.load(open(self.progressMLPpath))

    def load_bestMLP( self ):
        return cPickle.load(open(self.bestMLPpath))

    def training_monitor( self ):
        progressMLP = self.load_progressMLP()
        train_obj = numpy.array(progressMLP.monitor.channels['train_objective'].val_record)
        valid_obj = numpy.array(progressMLP.monitor.channels['valid_objective'].val_record)
        training_rate = numpy.array(progressMLP.monitor.channels['learning_rate'].val_record)
        #seconds_per_epoch = numpy.array(progressMLP.monitor.channels['seconds_per_epoch'].val_record)
        return (train_obj,valid_obj,training_rate)#,seconds_per_epoch)

    def training_graph( self ):
        train_obj, valid_obj, training_rate, = self.training_monitor()
        fig = plt.figure( figsize=(20,10))
        i = 1
        for start_at in [0, max(len(train_obj)-100,0)]:
            plt.subplot( 2, 2, i)
            i+=1
            plt.plot( range(start_at,len(train_obj)), train_obj[start_at:], color='b' )
            plt.plot( range(start_at,len(valid_obj)), valid_obj[start_at:], color='g' )
            plt.subplot( 2, 2, i)
            i+=1
            plt.plot( range(start_at+1,len(valid_obj)), valid_obj[start_at+1:]/valid_obj[start_at:-1], color='g' )
        plt.show()
#        return fig

    def mses( self ):
        bestMLP = cPickle.load(open(self.bestMLPpath))
        trainmse = numpy.array(bestMLP.monitor.channels['train_objective'].val_record[-1])
        validmse = numpy.array(bestMLP.monitor.channels['train_objective'].val_record[-1])
        return (trainmse,validmse)

    def train( self ):
        print self.yaml()
        train = self.parse_yaml()
        train.main_loop()
        
        self.generate()

    
    def generate( self, sigmacoeffs = [0,0.1,0.5,1.0], init_indices=[0,1] ):
        dataset = self.parse_yaml().dataset
        
        bestMLP = self.load_bestMLP()
        X = bestMLP.get_input_space().make_batch_theano()
        Y = bestMLP.fprop(X)
        pred_next_sample = theano.function( [X[0]], Y )
        
        print dir(dataset)
        init = dataset.get(['features'], init_indices)[0]
        
        init = numpy.tile( init, (len(sigmacoeffs),1) )
        print init.shape
        
        mrse = numpy.sqrt(self.mses()[1] )
        descs = map(lambda x: str(x[0]) + "-" + str(x[1]), itertools.product( sigmacoeffs, init_indices ) )
        sigmas = numpy.repeat( sigmacoeffs, len(init_indices) ).reshape( (8,1) ) * mrse
        print sigmas.shape
        
        print descs
        
        wave = numpy.zeros( (init.shape[0],32000) )
        wave[:,0:self.xsamples] = init
        for i in range(self.xsamples, wave.shape[1]-1, 1): #Generate waveform
            next_sample = pred_next_sample( wave[0:1,i-self.xsamples:i] )
            wave[:,i:i+1] = next_sample+numpy.random.normal( 0, 1, (init.shape[0],1) )*sigmas
        
        raw_wav = (wave*dataset._std + dataset._mean).astype( 'uint16' )
        
        for i in range(raw_wav.shape[0]):
            scipy.io.wavfile.write( self.pklprefix + "/" + self.string_desc + "-" + descs[i] + ".wav", rate=16000, data=raw_wav[i,:])

    def parse_yaml( self ):
        return yaml_parse.load(self.yaml())

    def yaml( self ):
        return """
        !obj:pylearn2.train.Train {
            dataset: &train !obj:research.code.pylearn2.datasets.timit.TIMITOnTheFly {
                which_set: 'train_train',
                frame_length: 1,
                frames_per_example: """ + str(self.xsamples )+ """,
                noise: """ + str(self.noise) + """,
                speaker_filter: [""" + str(self.speaker_id) + """],
                phone_filter: [""" + str(self.phone) + """],
                mid_third: True
            },
            model: !obj:mlp_with_source.MLPWithSource {
                batch_size: 100,
                layers: [
                    #!obj:mlp_with_source.CompositeLayerWithSource {
                    #layer_name: 'c',
                    #layers: [
                        !obj:pylearn2.models.mlp.RectifiedLinear {
                        layer_name: 'h1',
                        dim: """ + str(self.xsamples )+ """,
                        irange: """ + str(numpy.sqrt( 6.0/( 2*self.xsamples) ))  + """,
                        },
                    #],
                    #},
                    !obj:pylearn2.models.mlp.RectifiedLinear {
                        layer_name: 'h2',
                        dim: """ + str(self.xsamples )+ """,
                        irange: """ + str(numpy.sqrt( 6.0/( 2*self.xsamples ) )) + """,
                    },
                    !obj:pylearn2.models.mlp.Linear {
                        layer_name: 'o',
                        dim: 1,
                        irange: """ + str(numpy.sqrt( 6.0/ (self.xsamples + 1) )) + """
                    },
                ],
                input_space: !obj:pylearn2.space.CompositeSpace {
                    components: [
                        !obj:pylearn2.space.VectorSpace {
                            dim: """ + str(self.xsamples )+ """,
                        },
                    ],
                },
                input_source: ['features'],
            },
            algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
                learning_rate: """ + str(self.learnrate) + """,
                monitoring_dataset: {
                    'train': *train,
                    'valid': !obj:research.code.pylearn2.datasets.timit.TIMITOnTheFly {
                        which_set: 'train_valid',
                        frame_length: 1,
                        frames_per_example: """ + str(self.xsamples )+ """,
                        speaker_filter: [""" + str(self.speaker_id) + """],
                        phone_filter: [""" + str(self.phone) + """],
                        mid_third: True
                    },
                },
                cost: !obj:pylearn2.costs.cost.SumOfCosts {
                    costs: [
                        !obj:pylearn2.costs.mlp.Default {},
                        !obj:pylearn2.costs.mlp.WeightDecay {
                            coeffs: """ + str( [self.reg]*3 ) + """
                        }
                    ]
                },
            },
            extensions: [
                !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                    channel_name: 'valid_objective',
                    save_path: \"""" + self.bestMLPpath + """\",
                },
                !obj:pylearn2.training_algorithms.sgd.OneOverEpoch {
                    start: 2000,
                    half_life: 500
                }
                ],
            save_path: \"""" + self.progressMLPpath + """\",
            save_freq: 1
        }"""

if __name__=="__main__":
    print "Arguments: pklprefix learnrate reg speaker_id phone xsamples noise seed"
    whatdo = sys.argv[1]
    pklprefix = sys.argv[2]
    learnrate = float(sys.argv[3])
    reg = float(sys.argv[4])
    speaker_id = int(sys.argv[5])
    phone = int(sys.argv[6])
    noise = float(sys.argv[7])
    tm  = TrainedModel(pklprefix, learnrate = learnrate, reg=reg, speaker_id=speaker_id, phone=phone,noise=noise)
    if whatdo=='train' or whatdo=='forcetrain':
        if os.path.isfile(tm.progressMLPpath) and whatdo=='train':
            print "Are you sure you want to train?",tm.progressMLPpath, "exists. If so use forcetrain option."
        else:
            tm.train()
    elif whatdo=='monitor':
        tm.training_graph()
    elif whatdo=='showmse':
        mses = tm.mses()
        print "train", mses[0], "valid",mses[1]
    elif whatdo=='generate':
        tm.generate()
    elif whatdo=='yaml':
        print tm.yaml()
