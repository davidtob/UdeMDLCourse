import numpy
from string import Template
from pylearn2.config import yaml_parse
import matplotlib
matplotlib.use('agg')
import pylab as plt
import sys
import cPickle
import theano
import os
import scipy.io.wavfile
import itertools
import threading
import SimpleHTTPServer
import BaseHTTPServer
import logging
import io
import traceback
import os
import hashlib
import time
import base64
import urlparse

class TrainedModel(object):
    # This class should
    # 1. Be able to randomize hyper parameters
    # 2. store hyperparameters
    # 3. Produce yaml file
    # 4. Train based on yaml file
    # 5. Monitor training progress
    # 6. Compute MSE of trained model
    # 7. Generate from trained model
    def __init__(self, pklprefix=".", seed=0, learnrate = 0.0125, reg = 0.00005, xsamples=400, ysamples=1, noise="False", noise_decay=False ):
        self.pklprefix = pklprefix
        self.seed = seed
        numpy.random.seed( self.seed )
        self.learnrate = learnrate
        self.xsamples = xsamples
        self.ysamples = ysamples
        self.noise = noise
        self.noise_decay = noise_decay
        self.reg = reg
        numpy.random.seed( self.seed )

        self.trainlog = io.StringIO()
        
        self.string_desc = "%s-seed-%d-lr-%f-xs-%d-ys-%d-n-%s-nd-%s"%(self.string_desc_base,seed,learnrate,xsamples,ysamples,noise,str(noise_decay))
        self.progressMLPpath = self.pklprefix + "/progress-" + self.string_desc + ".pkl"
        self.bestMLPpath     = self.pklprefix + "/best-" + self.string_desc + ".pkl"
        
        self.MonitorServer = MonitorServer
    
    def load_progressMLP( self ):
        return cPickle.load(open(self.progressMLPpath))

    def load_bestMLP( self ):
        return cPickle.load(open(self.bestMLPpath))

    def training_monitor( self ):
        progressMLP = self.load_progressMLP()
        train_obj = numpy.array(progressMLP.monitor.channels['train_objective'].val_record)
        if 'valid_objective' in progressMLP.monitor.channels.keys():
            valid_obj = numpy.array(progressMLP.monitor.channels['valid_objective'].val_record)
        else:
            valid_obj = None
        training_rate = numpy.array(progressMLP.monitor.channels['learning_rate'].val_record)
        seconds_per_epoch = numpy.array(progressMLP.monitor.channels['training_seconds_this_epoch'].val_record)
        return (train_obj,valid_obj,training_rate,seconds_per_epoch)
    
    def epochs(self):
        if os.path.isfile(self.progressMLPpath):
            return len(self.training_monitor()[0])
        else:
            return 0

    def training_fig( self, secs_per_epoch=False ):
        import pylab as plt
        train_obj, valid_obj, training_rate,seconds_per_epoch = self.training_monitor()
        fig = plt.figure( figsize=(20,10))
        i = 1
        
        if secs_per_epoch==False:
            cols = 3
        else:
            cols = 4
        
        for start_at in [0, max(len(train_obj)-100,0)]:
            plt.subplot( 2, cols, i)
            i+=1
            plt.title( "train (blue) + valid(green) obj starting at %d"%start_at )
            plt.plot( range(start_at,len(train_obj)), train_obj[start_at:], color='b' )
            if valid_obj!=None:
                plt.plot( range(start_at,len(valid_obj)), valid_obj[start_at:], color='g' )
            plt.subplot( 2, cols, i)
            i+=1
            plt.title( "valid_obj improvement ratio at %d"%start_at )
            if valid_obj!=None:
                plt.plot( range(start_at+1,len(valid_obj)), valid_obj[start_at+1:]/valid_obj[start_at:-1] )
            else:
                plt.plot( range(start_at+1,len(train_obj)), train_obj[start_at+1:]/train_obj[start_at:-1] )
            plt.subplot( 2, cols, i)
            i+=1
            plt.title( "learning_rate" )
            plt.plot( range(start_at,len(training_rate)), training_rate[start_at:] )

            if start_at==0 and secs_per_epoch:
                plt.subplot( 2, cols, i)
                i+=1
                plt.title( "seconds per epoch" )
                plt.plot( range(len(seconds_per_epoch)), seconds_per_epoch )
            elif start_at>0 and secs_per_epoch:
                plt.subplot( 2, cols, i)
                i+=1
                plt.title( "seconds per epoch cumulative" )
                plt.plot( range(len(seconds_per_epoch)), numpy.cumsum(seconds_per_epoch)/3600.0 )
            
        return fig

    def mses( self ):
        try:
            bestMLP = cPickle.load(open(self.bestMLPpath))
        except:
            #print traceback.format_exc()
            return (0,0)
        trainmse = numpy.array(bestMLP.monitor.channels['train_objective'].val_record[-1])    
        if 'valid_objective' in bestMLP.monitor.channels.keys():
            validmse = numpy.array(bestMLP.monitor.channels['valid_objective'].val_record[-1])
        else:
            validmse = None
        return (trainmse,validmse)
        
    def rmses( self ):
        return map( lambda x: numpy.sqrt(x), self.mses() )

    def train( self ):
        self.start_web_server()
        
        root_logger = logging.getLogger( __name__ )# __name__.split('.')[0] )
        root_logger.addHandler( logging.StreamHandler( self.trainlog ) )
        root_logger.setLevel(logging.DEBUG)
        
        train = self.parse_yaml()
        train.main_loop()
        self.generate()
    
    def dataset_for_generation( self ): # If there is a validation use it for generation, since the training set might have noise added
        parsedyaml = self.parse_yaml()
        print parsedyaml.algorithm.monitoring_dataset
        if 'valid' in parsedyaml.algorithm.monitoring_dataset:
            dataset = parsedyaml.algorithm.monitoring_dataset['valid']
        else:
            dataset = parsedyaml.dataset
        return dataset
        
    def generate_pcm( self, sigmacoeffs = [0,0.1,0.5,1.0], init_indices=[0,1], length=32000 ):
        dataset = self.dataset_for_generation()
        if length==None:
            length = len(dataset.raw_wav[0])
        
        bestMLP = self.load_bestMLP()
        X = bestMLP.get_input_space().make_batch_theano()
        Y = bestMLP.fprop(X)
        pred_next_samples = theano.function( [X[0]], Y )
        
        init = dataset.get(['features'], init_indices)[0]
        init = numpy.tile( init, (len(sigmacoeffs),1) )

        trainmse, validmse = self.mses()
        if validmse!=None:
            mrse = numpy.sqrt( validmse )
        else:
            mrse = numpy.sqrt( trainmse )

        sigmas = numpy.repeat( sigmacoeffs, len(init_indices) ).reshape( ( len(sigmacoeffs)*len(init_indices),1 ) ) * mrse        
        
        wave = numpy.zeros( (init.shape[0],self.xsamples+((length-self.xsamples)/self.ysamples)*self.ysamples) )
        wave[:,0:self.xsamples] = init
        for i in range(self.xsamples, wave.shape[1]-1, self.ysamples): #Generate waveform
            next_sample = pred_next_samples( wave[:,i-self.xsamples:i] )
            wave[:,i:i+self.ysamples] = next_sample+numpy.random.normal( 0, 1, (init.shape[0],self.ysamples) )*sigmas
            print i
        
        raw_wav = (wave*dataset._std + dataset._mean).astype( 'int16' )
        return wave,raw_wav,dataset
    
    def generate( self, sigmacoeff = 0.1, init_index = 0, buf = None ):
        print "Generating"
        _,raw_wav = self.generate_pcm( [sigmacoeff], [init_index] )
        print "Done"
        if buf==None:
            buf = self.pklprefix + "/" + self.string_desc + "-" + descs[i] + ".wav"
        print buf, hasattr(buf,'write')
        scipy.io.wavfile.write( buf, rate=16000, data=raw_wav[0,:])

    #def generate_save( self, sigmacoeffs = [0,0.1,0.5,1.0], init_indices=[0,1] ):
    #    raw_wav = generate_wav( self, sigmacoeefs, init_indices )
    #    for i in range(raw_wav.shape[0]):
    #        scipy.io.wavfile.write( self.pklprefix + "/" + self.string_desc + "-" + descs[i] + ".wav", rate=16000, data=raw_wav[i,:])

    def parse_yaml( self ):
        return yaml_parse.load(self.yaml())
        
    def datasetyaml( self, trainorvalid ):
        raise "Not impelemented"
 
    def monitoringdatasetyaml( self ):
        raise "Not impelemented"
    
    def monitoringextensionyaml( self ):
        raise "Not implemented"
 
    def yaml( self ):
        return """
        !obj:pylearn2.train.Train {
            dataset: &train """ + self.datasetyaml('train') + """,
            model: !obj:mlp_with_source.MLPWithSource {
                batch_size: 100,
                layers: [
                    !obj:pylearn2.models.mlp.RectifiedLinear {
                        layer_name: 'h1',
                        dim: """ + str(self.xsamples )+ """,
                        irange: """ + str(numpy.sqrt( 6.0/( 2*self.xsamples) ))  + """,
                    },
                    !obj:pylearn2.models.mlp.RectifiedLinear {
                        layer_name: 'h2',
                        dim: """ + str(self.xsamples )+ """,
                        irange: """ + str(numpy.sqrt( 6.0/( 2*self.xsamples ) )) + """,
                    },
                    !obj:pylearn2.models.mlp.Linear {
                        layer_name: 'o',
                        dim: """ + str(self.ysamples) + """,
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
                monitoring_dataset: {""" + self.monitoringdatasetyaml() + """},
                cost: !obj:pylearn2.costs.cost.SumOfCosts {
                    costs: [
                        !obj:pylearn2.costs.mlp.Default {},
                        !obj:pylearn2.costs.mlp.WeightDecay {
                            coeffs: """ + str( [self.reg]*3 ) + """
                        }
                    ]
                },
                termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 3000
                }
            },
            extensions: [
                """ + self.monitoringextensionyaml() + """,
                !obj:pylearn2.training_algorithms.sgd.OneOverEpoch {
                    start: 1500,
                    half_life: 500
                }
                ],
            save_path: \"""" + self.progressMLPpath + """\",
            save_freq: 5
        }"""
    
    def start_web_server( self, wait = False ):
        def web_server_thread( cls ):
            httpd = None
            for port in range(8000, 8010):
                server_address = ('', port)
                try:
                    httpd = BaseHTTPServer.HTTPServer(server_address, self.MonitorServer)
                    break
                except:
                    print "Could not start on port",port,", trying next"
            assert httpd!=None
            httpd.RequestHandlerClass.tm = self
            sa = httpd.socket.getsockname()
            print "Serving HTTP on", sa[0], "port", sa[1], "..."
            httpd.serve_forever()
        
        t = threading.Thread( target = web_server_thread, args = (self,) )
        t.daemon = True
        t.start()
        if wait:
            try:
                while True:
                    time.sleep( 10000 )                
            except KeyboardInterrupt:
                return

class MonitorServer(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlparse.urlparse(self.path).path
        args = urlparse.parse_qs(urlparse.urlparse(self.path).query)
        if path=='/':
            commands = filter( lambda x: x[0:3]=="do_" and x!="do_HEAD" and x!="do_GET", dir(self) )
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'html')
            self.end_headers()
            self.wfile.write( "Available commands: " )
            map( lambda x: self.wfile.write(x[3:] + " "), commands )
        else:
            command = "do_" + path[1:]
            if command in dir(self):
                func = getattr( self, command )
                func(args)
            else:
                self.send_error(404, "File not found")
                return None     
    
    def do_trainlog(self, args):
        self.send_response(200, 'OK')
        self.send_header('Content-type', 'html')
        self.end_headers()
        print "log", self.tm.trainlog.getvalue()
        print "******************"
        self.wfile.write( self.tm.trainlog.getvalue() )
    
    def do_traingraph(self, args):
        if 'seconds' in args.keys():
            secs = True
        else:
            secs = False
        try:
            fig = self.tm.training_fig(secs)
        except:
            self.send_python_error()
        else:
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            buf = io.BytesIO()
            plt.savefig( buf )
            self.wfile.write( buf.getvalue() )
    
    def do_yaml(self, args):
        try:
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'html')
            self.end_headers()
            self.wfile.write( self.tm.yaml() )
        except:
            self.send_python_error()

    def do_bestmses(self, args):
        self.send_response(200, 'OK')
        self.send_header('Content-type', 'html')
        self.end_headers()
        trainmse,validmse = self.tm.mses()
        self.wfile.write( "train: %f "%trainmse )
        if validmse!=None:
            self.wfile.write( "valid: %f "%validmse )
    
    def do_generatewav(self, args):
        if 'sigma'in args.keys():
            print args['sigma'][0]
            sigma = float(args['sigma'][0])
        else:
            sigma = 0.1
        try:
            fn = hashlib.md5(str(hashlib.md5(str(time.time())))).hexdigest()
            wav = self.tm.generate( sigma, 0, fn )
            data = open(fn).read()
            os.remove(fn)
        except:
            self.send_python_error()
        else:
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'audio/vnd.wave')
            self.end_headers()
            self.wfile.write(data)
    
    def do_generatepcm(self, args):
        if 'sigmas' in args.keys():
            sigmas = map( lambda x: float(x), args['sigmas'][0].split(',') )
        else:
            sigmas = [0]
        if 'init_idx' in args.keys():
            init_idcs = [ int(args['init_idx'][0]) ]
        else:
            init_idcs = [0]
        try:
            arr = self.tm.generate_pcm( sigmas, init_idcs )
        except:
            self.send_python_error()
        else:
            self.send_ascii_encoded_array( arr )
    
    def send_ascii_encoded_array( self, arr ):
        self.send_response(200, 'OK')
        self.send_header('Content-type', 'html')
        self.end_headers()
        ascii = base64.b64encode( cPickle.dumps( arr ) )
        self.wfile.write( ascii )

    def send_python_error(self):
        self.send_response(200, 'OK')
        self.send_header('Content-type', 'html')
        self.end_headers()
        self.wfile.write( traceback.format_exc() )
        print traceback.format_exc()


#if __name__=="__main__":
#    print "Arguments: pklprefix learnrate reg speaker_id phone xsamples noise seed"
#    whatdo = sys.argv[1]
#    pklprefix = sys.argv[2]
#    learnrate = float(sys.argv[3])
#    reg = float(sys.argv[4])
#    speaker_id = int(sys.argv[5])
#    phone = int(sys.argv[6])
#    if sys.argv[7]=="False":
#      noise = False
#    else:
#      noise = float(sys.argv[7])
#    global tm
#    tm  = TrainedModel(pklprefix, learnrate = learnrate, reg=reg, speaker_id=speaker_id, phone=phone,noise=noise)
#    if whatdo=='train' or whatdo=='forcetrain':
#        if os.path.isfile(tm.progressMLPpath) and whatdo=='train':
#            print "Are you sure you want to train?",tm.progressMLPpath, "exists. If so use forcetrain option."
#        else:
#            tm.train()
#    elif whatdo=='server':
#        tm.start_web_server( True )
#    elif whatdo=='monitor':
#        tm.training_graph()
#    elif whatdo=='showmse':
#        mses = tm.mses()
#        print "train", mses[0], "valid",mses[1]
#    elif whatdo=='generate':
#        tm.generate()
#    elif whatdo=='yaml':
#        print tm.yaml()
