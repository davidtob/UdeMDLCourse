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
        self.framemax = framemax
        self.phonemax = phonemax
        self.sentmax = sentmax

        self.rescale = rescale
        if self.rescale=='framemax':
            Xnorm = Xraw/framemax
            ynorm = yraw/framemax
        elif self.rescale=='phonemax':
            Xnorm = Xraw/phonemax
            ynorm = yraw/phonemax
        elif self.rescale=='sentmax':
            Xnorm = Xraw/sentmax
            ynorm = yraw/sentmax
        elif self.rescale=='datasetmax':
            Xnorm = Xraw/max(phonemax)
            ynorm = yraw/max(phonemax)  
        else:
            raise Exception('Invalid rescaling option')
        
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
    
        if trainorvalid=='train':
            super(TimitPredFramesForPhn,self).__init__(X=self.Xfinal[shuffle_idxs[:num_train_examples],:], y=self.yfinal[shuffle_idxs[:num_train_examples],:])#, readnorm=False)
        else:
            super(TimitPredFramesForPhn,self).__init__(X=self.Xfinal[shuffle_idxs[num_train_examples:],:], y=self.yfinal[shuffle_idxs[num_train_examples:],:])#, readnorm=True)

class NormalizedData(DenseDesignMatrix):
    def __init__( self, X, y, readnorm=False ):
	if readnorm:
		Xmean, ymean, Xstd, ystd = cPickle.load( open("normalization","r" ) )
	else:
		Xmean = X.mean(0)
		ymean = y.mean(0)
		Xstd = X.std(0)
		ystd = y.std(0)
		cPickle.dump( (Xmean,ymean,Xstd,ystd), open("normalization","w+" ) )

	assert( sum(Xstd==0)==0 )
	assert( sum(ystd==0)==0 )

	X = (X - Xmean)/Xstd
	y = (y - ymean)/ystd

	super(NormalizedData,self).__init__(X=X, y=y)

class TimitPredFrames(DenseDesignMatrix):
    def __init__( self, xsamples, ysamples, sentences, train_valid_split, trainorvalid, seed, with_current_phone=False ):
	num_examples = 0
	for sent_idx in sentences:
		first_phn_start,last_phn_end = self.first_and_last_phn_offset( sent_idx )
		length = len(train.sentence_idx_to_wave( sent_idx ))
		num_examples += min(length-ysamples,last_phn_end) - max(first_phn_start,xsamples)
	
	Xsamp = np.zeros( (num_examples,xsamples) )
	Xphnonehot = np.zeros( (num_examples,61) )
	Xphndeltat = np.zeros( (num_examples,2) )
	y = np.zeros( (num_examples,ysamples) )

	phn_coding = numpy.eye(61) 

	print "constructing", num_examples, "examples"
	example_idx = 0		
	for sent_idx in sentences:
		print sent_idx
		for phn_start,phn_end,phn_num in train.phn[train.sentence_idx_to_phoneme_idcs(sent_idx)]:			
			for pred_sample in range(max(phn_start,xsamples),min(phn_end,length-ysamples)):
				#print a.shape, b.shape,Xsamp.shape,y.shape,start,min(phn_end,length-xsamples-ysamples),len(train.sentence_idx_to_wave( sent_idx ))
				Xsamp[example_idx,:], y[example_idx,:] = self.get_sample(sent_idx,pred_sample,xsamples,ysamples)
				Xphnonehot[example_idx,:] = phn_coding[phn_num]
				Xphndeltat[example_idx,0] = phn_start-pred_sample # time since phonenme start
				Xphndeltat[example_idx,1] = phn_end-pred_sample # time until phoneme ends
				example_idx = example_idx + 1

	print "done"

	#print X.mean()
	#print X.std()
	#print y.mean()
	# Normalize (sound signal should have mean zero)
	#std = y.std()
	#print y.std()
	print "sample standard deviation", y.std()
	Xsamp = Xsamp/y.std()
	y = y/y.std()

	print "Mean of phn delta t:", Xphndeltat.mean(0)
	print "Standard deviation of phn delta t:", Xphndeltat.std(0)
	Xphndeltat = (Xphndeltat - Xphndeltat.mean(0))/Xphndeltat.std(0)

	#X = Xsamp/(2.0**15)#std 
	#y = y/(2.0**15)#std

	if with_current_phone:
		X = numpy.hstack( (Xsamp, Xphnonehot, Xphndeltat) )
	else:
		X = Xsamp

	shuffle_idxs = numpy.random.permutation( num_examples )
	num_train_examples = int(numpy.floor(num_examples*train_valid_split))
	num_valid_examples = num_examples - num_train_examples

	if trainorvalid=='train':
		super(TimitPredFrames,self).__init__(X=X[shuffle_idxs[:num_train_examples],:], y=y[shuffle_idxs[:num_train_examples],:])#, readnorm=False)
	else:
		super(TimitPredFrames,self).__init__(X=X[shuffle_idxs[num_train_examples:],:], y=y[shuffle_idxs[num_train_examples:],:])#, readnorm=True)

    def get_sample( self, sent_idx, pred_sample, xsamples, ysamples ):
		wave = train.sentence_idx_to_wave( sent_idx )
		Xsamprow = wave[pred_sample-xsamples:pred_sample]
		yrow = wave[pred_sample:(pred_sample+ysamples)]
		#print yrow,(start+xsamples),(start+xsamples+ysamples),
		return Xsamprow,yrow

    def first_and_last_phn_offset( self, sent_idx ):
		first_phn = train.sentence_idx_to_phoneme_idcs(sent_idx)[0]
		last_phn = train.sentence_idx_to_phoneme_idcs(sent_idx)[-1]
		first_phn_start,_,_ = train.phn[first_phn]
		_,last_phn_end,_ = train.phn[last_phn]
		return first_phn_start,last_phn_end




class TimitRandomPredFrames(NormalizedData):
    def __init__( self, xsamples, ysamples, number_of_examples, seed=0,readnorm=False):
	print "Sampling random data"
	np.random.seed(seed)
	X = np.zeros( (number_of_examples,xsamples) )
	y = np.zeros( (number_of_examples,ysamples) )	
	start_time = time.clock()
	for i in range(number_of_examples):
		xrow, yrow = self.random_example( xsamples, ysamples )
		X[i,:] = xrow
		y[i,:] = yrow
		if time.clock()-start_time>5:
			print i
			start_time = time.clock()
	print "Done sampling"

        super(TimitRandomPredFrames,self).__init__(X=X, y=y, readnorm=readnorm)

    def random_example( self, xsamples, ysamples ):
	n = train.number_of_recorded_sentences()
	sent_idcs = np.random.permutation( range(n) )
	for sent_idx in sent_idcs:
		wave = train.sentence_idx_to_wave( sent_idx )
		wave = wave/float(max(wave))
		length = len(wave)
		if length >= xsamples+ysamples:
			start = int(np.floor(np.random.random()*float(length-xsamples-ysamples)))
			mid = start + xsamples
			end = start + xsamples + ysamples
			return (wave[start:mid], wave[mid:end])
	raise "There are no recorded sentences that are long enough"

    def load_cashed( self, xsamples, ysamples, seed ):
        try:
            f = open( "cache %d %d %d" % (xsamples, ysamples, seed) )
	except:
            return None
        X,y = cPickle.load(f)
        f.close()

    def write_cash( self, xsamples, ysamples, seed, X, y ):
        try:
            f = open( "cache %d %d %d" % (xsamples, ysamples, seed), "w" )
	except:
            return
        cPickle.write(f, (X,y) )
        f.close()

class TimitPhnWindow(NormalizedData):
    """
    Dataset for predicting the next acoustic sample.
    """
    def __init__( self, xsamples, ysamples, number_of_examples, seed=0, readnorm=False):

	self.phn_coding = numpy.eye(61) 
 
	print "Sampling random data"
	np.random.seed(seed)
	self.Xunnorm = np.zeros( (number_of_examples,xsamples + 61 + 2) )
	self.yunnorm = np.zeros( (number_of_examples,ysamples) )
	for i in range(number_of_examples):
		xrow, yrow = self.random_example( xsamples, ysamples )
		self.Xunnorm[i,:] = xrow
		self.yunnorm[i,:] = yrow
	print "Done sampling"

        super(TimitPhnWindow,self).__init__(X=self.Xunnorm, y=self.yunnorm,readnorm=readnorm)


    def random_example( self, xsamples, ysamples ):
	n = train.number_of_recorded_sentences()
	sent_idcs = np.random.permutation( range(n) )
	for sent_idx in sent_idcs:
		length = len(train.sentence_idx_to_wave( sent_idx ))
		if length >= xsamples+ysamples:
			while True:
				offset = int(np.floor(np.random.random()*float(length-xsamples-ysamples)))
				ret = self.one_example( xsamples, ysamples, sent_idx, offset )
				if not ret is None:
					return ret
	raise "There are no recorded sentences that are long enough"

    def one_example( self, xsamples, ysamples, sent_idx, offset ):
	wave = numpy.array(train.sentence_idx_to_wave( sent_idx )).astype(numpy.float32)
	mid = offset + xsamples
	end = offset + xsamples + ysamples
	X_samp = wave[offset:mid]
	y_samp = wave[mid:end]
	for phn_start,phn_end,phn in train.phn[train.sentence_idx_to_phoneme_idcs(sent_idx)]:
		if phn_start<=offset<phn_end:
			since_start = float(offset-phn_start)
			until_next = float(phn_end-offset)
			return numpy.concatenate( (X_samp,[since_start,until_next],self.phn_coding[phn,:]) ), y_samp
	#for phn_start,phn_end,phn in train.phn[train.sentence_idx_to_phoneme_idcs(sent_idx)]:
	#	print phn_start,phn_end,phn,offset,sent_idx
	#raise "bla"
	return None #The offset is not marked with a phoneme (happens at end of sentence 2789)

    def load_cashed( self, xsamples, ysamples, seed ):
        try:
            f = open( "cache %d %d %d" % (xsamples, ysamples, seed) )
	except:
            return None
        X,y = cPickle.load(f)
        f.close()

    def write_cash( self, xsamples, ysamples, seed, X, y ):
        try:
            f = open( "cache %d %d %d" % (xsamples, ysamples, seed), "w" )
	except:
            return
        cPickle.write(f, (X,y) )
        f.close()




#a = timit_data.TimitRandomPredFrames( 1000, 1000, 10 )
	

class TimitPhoneData(DenseDesignMatrix):
    """
    Dataset with frames and corresponding one-hot encoded
    phones.
    """
    def __init__(self, datapath, framelen, overlap, start=0, stop=None):
        """
        datapath: path to TIMIT raw data (using WAV format)
        framelen: length of the acoustic frames
        overlap: amount of acoustic samples to overlap
        start: index of first TIMIT file to be used
        end: index of last TIMIT file to be used
        """
        data = TimitFullCorpusReader(datapath)
        # Some list comprehension/zip magic here (but it works!)
        if stop is None:
            utterances = data.utteranceids()[start:]
        else:
            utterances = data.utteranceids()[start:stop]
        spkrfr = [data.frames(z, framelen, overlap) for z in
                  utterances]
        fr, ph = zip(*[(x[0], x[1]) for x in spkrfr])
        X = np.vstack(fr)*2**-15
        ph = list(itertools.chain(*ph))

        # making y a one-hot output
        one_hot = np.zeros((len(ph),len(data.phonelist)),dtype='float32')
        idx = [data.phonelist.index(p) for p in ph]
        for i in xrange(len(ph)):
            one_hot[i,idx[i]] = 1.
        y = one_hot

        super(TimitPhoneData,self).__init__(X=X, y=y)
