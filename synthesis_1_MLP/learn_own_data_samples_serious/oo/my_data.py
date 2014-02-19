from pylearn2.datasets import DenseDesignMatrix
#import numpy as np
import numpy
import scipy.io.wavfile

class PredFrames(DenseDesignMatrix):
    def __init__( self, xsamples, ysamples, recenter_samples, rescaling, phone, max_examples, trainorvalid, train_valid_split, seed ):
        numpy.random.seed( seed )
        if phone=='aa':
            rate,wave1 = scipy.io.wavfile.read("../../own data/Aa.wav")
            wave1 = wave1[2*16000:-16000]
            rate,wave2 = scipy.io.wavfile.read("../../own data/Aa2.wav")
            wave2 = wave2[2*16000:-16000]
            waves = [wave1,wave2]
            #waves = [ [0,1,2,3,4,5,6,7,8,9,10]]
        elif phone=='oo':
            rate,wave1 = scipy.io.wavfile.read("../../own data/Oo.wav")
            wave1 = wave1[2*16000:-16000]
            rate,wave2 = scipy.io.wavfile.read("../../own data/Oo2.wav")
            wave2 = wave2[2*16000:-16000]
            waves = [wave1,wave2]            
        else:
            raise "No such phone"
        
        num_examples = sum(map( lambda x: len(x), waves )) - len(waves)*(xsamples+ysamples)
        num_examples = min(num_examples,max_examples)
        
        self.Xraw = numpy.zeros( (num_examples,xsamples) )
        self.yraw = numpy.zeros( (num_examples,ysamples) )
        
        print "constructing examples"
        example_idx = 0
        for wave in waves:
            if len(wave) > xsamples+ysamples:
                # 0 1 2 3 4 5 6
                # 0 1 2   3 4       
                # 1 2 3   4 5
                # 2 3 4   5 6
                for i in range(len(wave)-xsamples-ysamples):
                    self.Xraw[example_idx,:] = wave[i:i+xsamples]
                    self.yraw[example_idx,:] = wave[i+xsamples:i+xsamples+ysamples]
                    example_idx = example_idx + 1
                    if example_idx >= num_examples:
                        break
                if example_idx >= num_examples:
                    break
            print "Done constructing"
        
        self.recenter_samples = recenter_samples
        if self.recenter_samples:
            self.mu = numpy.mean( [sample for wave in waves for sample in wave ] )
            self.Xrecent = self.Xraw - self.mu
            self.yrecent = self.yraw - self.mu
        else:
            self.Xrecent = self.Xraw
            self.yrecent = self.yraw
        del self.Xraw
        del self.yraw
        
        self.rescaling = rescaling
        if rescaling=='empirical_std':
            self.std = numpy.std( [sample for wave in waves for sample in wave ] )
            self.Xrescale = self.Xrecent/self.std
            self.yrescale = self.yrecent/self.std
        elif rescaling=='samples_max':
            self.max =  numpy.max( [sample for wave in waves for sample in wave ] )
            self.Xrescale = self.Xrecent/self.max
            self.yrescale = self.yrecent/self.max
        elif rescaling=='2**15':
            self.Xrescale = self.Xrecent/(2**15)
            self.yrescale = self.yrecent/(2**15)
        else:
            raise rescaling + " not a valid option for rescaling"
        del self.Xrecent
        del self.yrecent
            
#        self.normalize_component = normalize_component
#        if self.normalize_component:
#            self.comp_stds = 
        
        self.Xfinal = self.Xrescale
        self.yfinal = self.yrescale
        #std = X[:,0].std()
        #print "sample standard deviation", std
        #X = X/std
        #y = y/std
        
        #print X.shape
        #print y.shape
        #num_examples = X.shape[0]
        self.shuffle_idxs = numpy.random.permutation( num_examples )
        print num_examples,train_valid_split
        num_train_examples = int(numpy.floor(num_examples*train_valid_split))
        num_valid_examples = num_examples - num_train_examples
        
        if trainorvalid=='train':
            super(PredFrames,self).__init__(X=self.Xfinal[self.shuffle_idxs[:num_train_examples],:], y=self.yfinal[self.shuffle_idxs[:num_train_examples],:])#, readnorm=False)
        else:
            super(PredFrames,self).__init__(X=self.Xfinal[self.shuffle_idxs[num_train_examples:],:], y=self.yfinal[self.shuffle_idxs[num_train_examples:],:])#, readnorm=True)

    def invert( self, gen ):
        ret = []
        for yrescale in gen:
            if self.rescaling=='empirical_std':
                yrecent = yrescale*self.std
            elif self.rescaling=='samples_max':
                yrecent = yrescale*self.max
            elif self.rescaling=='2**15':
                yrecent = yrescale*(2**15)
            else:
                raise rescaling + " not a valid option for rescaling"

            if self.recenter_samples==True:
                yraw = yrecent + self.mu
            else:
                yraw = yrecent
            ret.append( yraw )
        return ret
               
#    def get_sample( self, wave, pred_sample, xsamples, ysamples ):
#        Xsamprow = wave[pred_sample-xsamples:pred_sample]
#        yrow = wave[pred_sample:(pred_sample+ysamples)]
#        #print yrow,(pred_sample+xsamples),(pred_sample+xsamples+ysamples),
#        return Xsamprow,yrow

class PredFramesDFT(DenseDesignMatrix):
    def __init__( self, framelen, Xframes, yframes, representation, normalize_mag, normalize_component, phone, max_examples, train_valid_split, trainorvalid, seed ):
        if phone=='aa':
            rate,wave1 = scipy.io.wavfile.read("../own data/Aa.wav")
            wave1 = wave1[2*16000:-16000]
            rate,wave2 = scipy.io.wavfile.read("../own data/Aa2.wav")
            wave2 = wave2[2*16000:-16000]
            waves = [wave1,wave2]
        else:
            raise "No such phone"
        
        self.representation = representation
        self.normalize_mag = normalize_mag
        self.normalize_component = normalize_component
        assert( representation=='real_imag' or representation=='mag_arg' )

    
        num_examples = sum(map( lambda x: len(x), waves )) - len(waves)*(framelen*(Xframes+yframes))
        num_examples = min(num_examples,max_examples)
     
        self.framelen = framelen
        self.rfft_len = rfft_len = framelen/2 + 1
                    
        self.frames = numpy.zeros( (num_examples+Xframes+yframes-1, rfft_len), dtype=numpy.complex64 )
        # 0 1 2
        # 2 3 4
        # 1 2 3
     
        print "constructing examples"
        frame_idx = 0
        for wave in waves:
            if len(wave) > framelen*(Xframes + yframes):
                for frame_start in range(0,len(wave)-framelen*yframes,framelen):
                    frame = wave[frame_start:frame_start+framelen]
                    frame_dft = numpy.fft.rfft( frame )
                    self.frames[frame_idx,:] = frame_dft
                    frame_idx = frame_idx + 1
                    if frame_idx >= self.frames.shape[0]:
                        break
                if frame_idx >= frame_idx >= self.frames.shape[0]:
                    break
            print "Done constructing"
        
        if normalize_mag==True:
            self.mag_mean = numpy.abs(self.frames).mean(0)
            self.framesmagnorm = self.frames/self.mag_mean
        else:
           self.framesmagnorm = self.frames

        if representation=='real_imag':
            self.framesrepr = numpy.hstack( (numpy.real(self.framesmagnorm),numpy.imag(self.framesmagnorm) ) )
        elif representation=='mag_arg':
            self.framesrepr = numpy.hstack( (numpy.abs(self.framesmagnorm),numpy.angle(self.framesmagnorm) ) )
        
        if normalize_component==True:
            self.std = self.framesrepr.std(0)
            self.mean = self.framesrepr.mean(0)        
            #print "standard deviation", self.std
            #print "mean", self.mean            
            self.framescompnorm = (self.framesrepr-self.mean)/(self.std + (self.std==0)) # Why is last fourrier coeff constant
        else:
            self.framescompnorm = self.framesrepr

        # Extract the correct frames for each training example        
        self.Xrowindices = map( lambda x: range(x,x+Xframes), range(num_examples)  )
        self.Xrows = self.framescompnorm[ self.Xrowindices,:]
        self.Xrowsreshaped = map( lambda x: x.reshape( (1,Xframes*rfft_len*2) ), self.Xrows )
        self.Xall = numpy.vstack( self.Xrowsreshaped )

        self.yrowindices = map( lambda x: range(x+Xframes,x+Xframes+yframes), range(num_examples)  )
        self.yrows = self.framescompnorm[ self.yrowindices,:]
        self.yrowsreshaped = map( lambda x: x.reshape( (1,yframes*rfft_len*2) ), self.yrows )
        self.yall = numpy.vstack( self.yrowsreshaped )
            
        shuffle_idxs = numpy.random.permutation( num_examples )
        num_train_examples = int(numpy.floor(num_examples*train_valid_split))
        num_valid_examples = num_examples - num_train_examples
    
        if trainorvalid=='train':
            super(PredFramesDFT,self).__init__(X=self.Xall[shuffle_idxs[:num_train_examples],:], y=self.yall[shuffle_idxs[:num_train_examples],:])#, readnorm=False)
        else:
            super(PredFramesDFT,self).__init__(X=self.Xall[shuffle_idxs[num_train_examples:],:], y=self.yall[shuffle_idxs[num_train_examples:],:])#, readnorm=True)

    def invert( self, framescompnorm ):
        print "0:", framescompnorm[0,0:2]

        if self.normalize_component:
            framesrepr = framescompnorm*(self.std + (self.std==0)) + self.mean
        else:
            framesrepr = framescompnorm
        
        print "1:", framesrepr[0,0:2]


        if self.representation=='real_imag':
            frames_dft_real = framesrepr[:,0:self.rfft_len]         
            frames_dft_imag = framesrepr[:,self.rfft_len:2*self.rfft_len]
            framesmagnorm = frames_dft_real + 1j*frames_dft_imag
        elif self.representation=='mag_arg':
            frames_dft_mag  = framesrepr[:,0:self.rfft_len]         
            frames_dft_arg = framesrepr[:,self.rfft_len:2*self.rfft_len]
            framesmagnorm = frames_dft_mag*exp( 1j*frames_dft_arg )
        
        print "2:",framesmagnorm[0,0:2]

        if self.normalize_mag:
            frames = framesmagnorm*self.mag_mean
        else:
            frames = framesmagnorm
        
        print "3:",frames[0,0:2]
            
        wave = []
        for frame_dft in frames:
            frame_wave = numpy.fft.irfft( frame_dft )
            wave = numpy.hstack( (wave,frame_wave) )
        return wave
        
#a = PredFramesDFT( 128, 2, 1, 'mag_arg', False, False, 'aa', 3, 0.8, 'train', 0)
#a = PredFramesDFT( 128, 1, 1, 'mag_arg', False, True, 'aa', 1000, 0.8, 'train', 0)
#a = PredFramesDFT( 128, 1, 1, 'mag_arg', True, False, 'aa', 1000, 0.8, 'train', 0)
#a = PredFramesDFT( 128, 1, 1, 'mag_arg', True, True, 'aa', 1000, 0.8, 'train', 0)
#print a.frames[0,:2]
#print a.invert( a.framescompnorm[0:1,:] )[0:2]
#print wave1[0:2]
