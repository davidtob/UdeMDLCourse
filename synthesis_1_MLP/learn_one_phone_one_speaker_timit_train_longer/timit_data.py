#  10/2 : Downloaded this file from Joaos github
from pylearn2.datasets import DenseDesignMatrix
#from timit_full import TimitFullCorpusReader
import numpy as np
import itertools
import time
#import timit_full
from load_data import *

#execfile('../../load_data.py')
#execfile('~/synt/../load_data.py')

class TimitPredFramesForPhn(DenseDesignMatrix):
    def __init__( self, xsamples, ysamples, rescale, phone, num_examples, train_valid_split, trainorvalid, seed, speaker_id = None, speaker_info = False ):
        Xraw = np.zeros( (num_examples,xsamples) )
        yraw = np.zeros( (num_examples,ysamples) )
        speakerraw = np.zeros( (num_examples, len(spkrinfo[0])))
        framemax = np.zeros( (num_examples,1))
        phonemax = np.zeros( (num_examples,1))
        sentmax = np.zeros( (num_examples,1))
        
        self.speaker_id = speaker_id
        
        t = time.time()
        print "constructing frames", len(train.phn)
        example_idx = 0
        for idx in range(len(train.phn)):
            if time.time()>t+5:
                t = time.time()
                print "Have",example_idx,"examples constructed"
            if train.phoneme_idx_to_phoneme_str(idx)==phone:
                sent_idx = train.phoneme_idx_to_sentence_idx(idx)
                if train.spkr[sent_idx]==speaker_id or speaker_id==None:
                    start, end = train.phoneme_idx_to_offsets(idx)
                    length = end-start
                    wave = train.sentence_idx_to_wave(sent_idx)
                    wavemax = max( wave )
                    start2 = start + length/3
                    end2 = end - length/3
                    phnmax =  max( wave[ start2:end2 ] )
                    if end2-start2 > xsamples+ysamples:
                        for i in range(start2,end2-xsamples-ysamples):
                            Xraw[example_idx,:] = wave[ i:(i+xsamples) ]
                            yraw[example_idx,:] = wave[ (i+xsamples):(i+xsamples+ysamples) ]
                            speakerraw[example_idx,:] = spkrinfo[train.spkr[sent_idx]]
                            framemax[example_idx,:] = max( wave[ i:(i+xsamples+ysamples) ] )
                            phonemax[example_idx,:] = phnmax
                            sentmax[example_idx,:] = wavemax
                            example_idx += 1
                            if example_idx >= num_examples:
                                break
                    if example_idx >= num_examples:
                        break
        
        if example_idx < num_examples:
            num_examples = example_idx-1
            Xraw = Xraw[0:num_examples,:]
            yraw = yraw[0:num_examples,:]
            speakerraw = speakerraw[0:num_examples,:]
            framemax = framemax[0:num_examples,:]
            phonemax = phonemax[0:num_examples,:]
            sentmax = sentmax[0:num_examples,:]
        
        print "Done constructing",num_examples,"examples"

        self.Xraw = Xraw
        self.yraw = yraw
                
        #self.framemax = framemax
        #self.phonemax = phonemax
        #self.sentmax = sentmax

        self.rescale = rescale
        if self.rescale=='framemax':
            self.scaling = framemax
        elif self.rescale=='phonemax':
            self.scaling = phonemax
        elif self.rescale=='sentmax':
            self.scaling = sentmax
        elif self.rescale=='datasetmax':
            self.scaling = numpy.ones( phonemax.shape )*max(phonemax)
        else:
            raise Exception('Invalid rescaling option')

        Xnorm = Xraw/self.scaling
        ynorm = yraw/self.scaling
        
        self.age_norm = (speakerraw[:,0].mean(), speakerraw[:,0].std())
        self.height_norm =(speakerraw[:,15].mean(), speakerraw[:,15].std())
        speakernorm = speakerraw
        speakernorm[:,0] = (speakernorm[:,0] - self.age_norm[0])/self.age_norm[1]
        speakernorm[:,15] = (speakernorm[:,15] - self.height_norm[0])/self.height_norm[1]
        
        self.yfinal = ynorm
        self.speaker_info = speaker_info
        if self.speaker_info:
            self.Xfinal = numpy.hstack( (Xnorm,speakernorm) )
        else:
            self.Xfinal = Xnorm
        
        np.random.seed(seed)
        shuffle_idxs = numpy.random.permutation( num_examples )
        self.shuffle_idxs = shuffle_idxs
        num_train_examples = int(numpy.floor(num_examples*train_valid_split))
        num_valid_examples = num_examples - num_train_examples
        self.num_valid_examples = num_valid_examples
        self.num_train_examples = num_train_examples
    
        X,y = self.getset(trainorvalid)
        super(TimitPredFramesForPhn,self).__init__(X=X, y=y)
    
    def getset( self, trainorvalid ):
        if trainorvalid=='train':
            idxs = self.shuffle_idxs[:self.num_train_examples]
        elif trainorvalid=='valid':
            idxs = self.shuffle_idxs[self.num_train_examples:]
        else:
            raise "Trainordvalid must be 'train' or 'valid'"
        return (self.Xfinal[idxs,:],self.yfinal[idxs,:])
    
    def invert( self, i, y ):
        # Take a generated ys and convert back to original scale, using scaling of example i
        return y*self.scaling[i]
#        if self.rescale=='framemax':
#            return y*self.framemax[i]
#        elif self.rescale=='phonemax':
#            return y*self.phonemax[i]			
#        elif self.rescale=='sentmax':
#            return y*self.sentmax[i]
#        elif self.rescale=='datasetmax':
#            return y*max(self.phonemax)
    
    def mse( self, trainorvalid, ypred ):
        if trainorvalid=='train':
            i = self.shuffle_idxs[:self.num_train_examples]
        else:
            i = self.shuffle_idxs[self.num_train_examples:]        
        # Compute MSE
#        if self.rescale=='framemax':
#            ypredraw = ypred*self.framemax[i]
#        elif self.rescale=='phonemax':
#            ypredraw = ypred*self.phonemax[i]
#        elif self.rescale=='sentmax':
#            ypredraw = ypred*self.sentmax[i]
#        elif self.rescale=='datasetmax':
#            ypredraw = ypred*max(self.phonemax)
        ypredraw = ypred * self.scaling[i]
                    
        return sum(sum((ypredraw-self.yraw[i,:])**2))/(ypred.shape[0]*ypred.shape[1])

    def dataset_std( self ):
        tot = 0
        n = 0
        for i in range(train.number_of_recorded_sentences()):
            tot = tot + sum( train.sentence_idx_to_wave(i).astype(numpy.float32)**2 )
            n = n + len(train.sentence_idx_to_wave(i))
        return numpy.sqrt(tot/n)
     

