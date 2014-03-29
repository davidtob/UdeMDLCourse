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
    def __init__(self, pklprefix=".", seed=0, learnrate = 0.0125, reg = 0.00005, xsamples=400, noise=False ):
        self.pklprefix = pklprefix
        self.seed = seed
        numpy.random.seed( self.seed )
        self.learnrate = learnrate
        self.xsamples = xsamples
        self.noise = noise
        self.reg = reg
        numpy.random.seed( self.seed )

        self.trainlog = io.StringIO()
        
        self.string_desc = "%s-%d-%f-%d-%s"%(self.string_desc_base,seed,learnrate,xsamples,str(noise))
        self.progressMLPpath = self.pklprefix + "/progress-" + self.string_desc + ".pkl"
        self.bestMLPpath     = self.pklprefix + "/best-" + self.string_desc + ".pkl"
    
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
        #seconds_per_epoch = numpy.array(progressMLP.monitor.channels['seconds_per_epoch'].val_record)
        return (train_obj,valid_obj,training_rate)#,seconds_per_epoch)

    def training_fig( self ):
        import pylab as plt
        train_obj, valid_obj, training_rate, = self.training_monitor()
        fig = plt.figure( figsize=(20,10))
        i = 1
            
        for start_at in [0, max(len(train_obj)-100,0)]:
            plt.subplot( 2, 3, i)
            i+=1
            plt.title( "train+valid obj starting at %d"%start_at )
            plt.plot( range(start_at,len(train_obj)), train_obj[start_at:], color='b' )
            if valid_obj!=None:
                plt.plot( range(start_at,len(valid_obj)), valid_obj[start_at:], color='g' )
            plt.subplot( 2, 3, i)
            i+=1
            plt.title( "valid_obj improvement ratio at %d"%start_at )
            if valid_obj!=None:
                plt.plot( range(start_at+1,len(valid_obj)), valid_obj[start_at+1:]/valid_obj[start_at:-1] )
            else:
                plt.plot( range(start_at+1,len(train_obj)), train_obj[start_at+1:]/train_obj[start_at:-1] )
            plt.subplot( 2, 3, i)
            i+=1
            plt.title( "learning_rate" )
            plt.plot( range(start_at,len(training_rate)), training_rate[start_at:] )
        return fig

    def mses( self ):
        bestMLP = cPickle.load(open(self.bestMLPpath))
        trainmse = numpy.array(bestMLP.monitor.channels['train_objective'].val_record[-1])
        validmse = numpy.array(bestMLP.monitor.channels['valid_objective'].val_record[-1])
        return (trainmse,validmse)

    def train( self ):
        self.start_web_server()
        
        root_logger = logging.getLogger( __name__ )# __name__.split('.')[0] )
        root_logger.addHandler( logging.StreamHandler( self.trainlog ) )
        root_logger.setLevel(logging.DEBUG)
        
        train = self.parse_yaml()
        train.main_loop()
        self.generate()
    
    def generate_pcm( self, sigmacoeffs = [0,0.1,0.5,1.0], init_indices=[0,1], length=32000 ):
        dataset = self.parse_yaml().dataset
        if length==None:
            length = len(dataset.raw_wav[0])
        
        bestMLP = self.load_bestMLP()
        X = bestMLP.get_input_space().make_batch_theano()
        Y = bestMLP.fprop(X)
        pred_next_sample = theano.function( [X[0]], Y )
        
        init = dataset.get(['features'], init_indices)[0]
        print init
        
        init = numpy.tile( init, (len(sigmacoeffs),1) )
        print init

        if sigmacoeffs!=[0]:
            mrse = numpy.sqrt( self.mses()[1] )
        else:
            mrse = numpy.zeros( (len(init_indices),1) )

        descs = map(lambda x: str(x[0]) + "-" + str(x[1]), itertools.product( sigmacoeffs, init_indices ) )
        sigmas = numpy.repeat( sigmacoeffs, len(init_indices) ).reshape( ( len(sigmacoeffs)*len(init_indices),1 ) ) * mrse
        
        wave = numpy.zeros( (init.shape[0],length) )
        wave[:,0:self.xsamples] = init
        for i in range(self.xsamples, wave.shape[1]-1, 1): #Generate waveform
            next_sample = pred_next_sample( wave[0:1,i-self.xsamples:i] )
            wave[:,i:i+1] = next_sample+numpy.random.normal( 0, 1, (init.shape[0],1) )*sigmas
        
        raw_wav = (wave*dataset._std + dataset._mean).astype( 'int16' )
        print wave[0,0:10]
        print raw_wav[0,0:10]
        print dataset._std
        print dataset._mean
        return raw_wav,dataset
    
    def generate( self, sigmacoeff = 0.1, init_index = 0, buf = None ):
        print "Generating"
        raw_wav = self.generate_pcm( [sigmacoeff], [init_index] )
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
                monitoring_dataset: {""" + self.monitoringdatasetyaml() + """},
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
                """ + self.monitoringextensionyaml() + """,
                !obj:pylearn2.training_algorithms.sgd.OneOverEpoch {
                    start: 2000,
                    half_life: 500
                }
                ],
            save_path: \"""" + self.progressMLPpath + """\",
            save_freq: 1
        }"""
    
    def start_web_server( self, wait = False ):
        def web_server_thread( cls ):
            httpd = None
            for port in range(8000, 8010):
                server_address = ('', port)
                try:
                    httpd = BaseHTTPServer.HTTPServer(server_address, MonitorServer)
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
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'html')
            self.end_headers()
            self.wfile.write( 'Available commands: trainlogstdout trainlogstderr traingraph yaml generatewav generatepcm' )
        elif path=='/trainlogstdout':
            self.do_trainlog(0)
        elif path=='/trainlogstderr':
            self.do_trainlog(1)
        elif path=='/traingraph':
            self.do_traingraph()
        elif path=='/yaml':
            self.do_yaml()
        elif path=='/generatewav':
            self.do_generatewav(args)
        elif path=='/generatepcm':
            self.do_generatepcm()
        else:
            self.send_error(404, "File not found")
            return None
    
    def do_trainlog(self, i):
        self.send_response(200, 'OK')
        self.send_header('Content-type', 'html')
        self.end_headers()
        print "log", self.tm.trainlog.getvalue()
        print "******************"
        self.wfile.write( self.tm.trainlog.getvalue() )
    
    def do_traingraph(self):
        try:
            fig = self.tm.training_fig()
        except:
            self.do_error()
        else:
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            buf = io.BytesIO()
            plt.savefig( buf )
            self.wfile.write( buf.getvalue() )
    
    def do_yaml(self):
        self.send_response(200, 'OK')
        self.send_header('Content-type', 'html')
        self.end_headers()
        self.wfile.write( self.tm.yaml() )
    
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
            self.do_error()
        else:
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'audio/vnd.wave')
            self.end_headers()
            self.wfile.write(data)
    
    def do_generatepcm(self):
        try:
            arr = self.tm.generate_pcm( [0], [0] )
        except:
            self.do_error()
        else:
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'html')
            self.end_headers()
            ascii = base64.b64encode( cPickle.dumps( arr ) )
            self.wfile.write( ascii )

    def do_error(self):
        self.send_response(200, 'OK')
        self.send_header('Content-type', 'audio/vnd.wave')
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
