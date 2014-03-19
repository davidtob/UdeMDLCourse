import numpy
from string import Template
from pylearn2.config import yaml_parse
import pylab as plt
import sys

class TrainedModel:
    # This class should
    # 1. Be able to randomize hyper parameters
    # 2. store hyperparameters
    # 3. Produce yaml file
    # 4. Train based on yaml file
    # 5. Monitor training progress
    # 6. Compute MSE of trained model
    # 7. Generate from trained model
    def __init__(self, pklprefix, learnrate = 0.6, reg = 0.00005, speaker_id=104, phone=28, xsamples=400, noise=0.1, seed=0 ):
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

    def training_monitor( self ):
        progressMLP = cPickle.load(open(self.progressMLPpath))
        train_obj = numpy.array(progressMLP.monitor.channels['train_objective'].val_record)
        valid_obj = numpy.array(progressMLP.monitor.channels['valid_objective'].val_record)
        training_rate = numpy.array(progressMLP.monitor.channels['training_rate'].val_record)
        seconds_per_epoch = numpy.array(progressMLP.monitor.channels['seconds_per_epoch'].val_record)
        return (train_obj,valid_obj,training_rate,seconds_per_epoch)

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
        return fig

    def mse( self ):
        bestMLP = cPickle.load(open(self.bestMLPpath))
        trainmse = numpy.array(bestMLP.monitor.channels['train_objective'].val_record[-1])
        validmse = numpy.array(progressMLP.monitor.channels['train_objective'].val_record[-1])
        return (trainmse,validmse)

    def generate( self ):
        pass

    def train( self ):
        print self.yaml()
        train = yaml_parse.load(self.yaml())
        train.main_loop()

    def yaml( self ):
        print self.progressMLPpath
        return """
        !obj:pylearn2.train.Train {
            dataset: &train !obj:research.code.pylearn2.datasets.timit.TIMIT {
                which_set: 'train',
                frame_length: 1,
                frames_per_example: """ + str(self.xsamples )+ """,
                noise: """ + str(self.noise) + """,
                speaker_filter: [""" + str(self.speaker_id) + """],
                phone_filter: [""" + str(self.phone) + """],
            },
            model: !obj:mlp_with_source.MLPWithSource {
                batch_size: 512,
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
                    'valid': !obj:research.code.pylearn2.datasets.timit.TIMIT {
                        which_set: 'valid',
                        frame_length: 1,
                        frames_per_example: """ + str(self.xsamples )+ """,
                        speaker_filter: [""" + str(self.speaker_id) + """],
                        phone_filter: [""" + str(self.phone) + """],
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
    #print "Arguments: pklprefix learnrate reg speaker_id phone xsamples noise seed"
    pklprefix = sys.argv[0]
    #learnrate = sys.argv[1]
    #reg       = sys.arg
    tm  = TrainedModel(pklprefix)
    tm.train()
