import os
import numpy
import gc
import sys

timit_concat = os.environ["PYLEARN2_DATA_PATH"] + "/timit/readable/timit_concat"
#train_wav = numpy.memmap( timit_concat + "/train_wav.npy", dtype='int', mode='r' )
print "train",
print "loading wav",
sys.stdout.flush()
train_wav = numpy.load( timit_concat + "/train_wav.npy" )
print train_wav.dtype
print "words",
sys.stdout.flush()
train_words = numpy.load( timit_concat + "/train_words.npy" )
print train_words.dtype
print "phones",
sys.stdout.flush()
train_phones = numpy.load( timit_concat + "/train_phones.npy" )
print "phonemes",
sys.stdout.flush()
train_phonemes = numpy.load( timit_concat + "/train_phonemes.npy" )
print "intervals",
sys.stdout.flush()
train_intervals = numpy.load( timit_concat + "/train_intervals.npy" )
print ""
#print "valid",
#print "loading wav",
#sys.stdout.flush()
#valid_wav = numpy.load( timit_concat + "/valid_wav.npy" )
#print "words",
#sys.stdout.flush()
#valid_words = numpy.load( timit_concat + "/valid_words.npy" )
#print "phones",
#sys.stdout.flush()
#valid_phones = numpy.load( timit_concat + "/valid_phones.npy" )
#print "phonemes",
#sys.stdout.flush()
#valid_phonemes = numpy.load( timit_concat + "/valid_phonemes.npy" )
#print "intervals",
#sys.stdout.flush()
#valid_intervals = numpy.load( timit_concat + "/valid_intervals.npy" )
#print ""

print len(train_wav)
print train_intervals.shape

num_examples = train_intervals[-1] - 400*len(train_intervals)

#def next( subset ):



#def complete_subset( elements ):
#    subset[0] = [ 2**32 ]
#    subset[1] = [ 2**16, 2**16 ]
#    subset[2] = [ 2**16, 2**16 ]

#def random_subset( n, k ):
#    ret = numpy.array( n/32 + 1, dtype='uint32' )
#    for i in range(n):
#        ret[ i/32 ] 

subset = [0xFFFFFFFFFFFFFFFF]*(2**32/64)

def num_bits( n ):
    # Number of bits in 64 bit number
    ret = 0
    a = 1
    for i in range(64):
        ret = ret + ((n&a)>0)
        a = 2*a
    return ret

#while True:
#    print num_bits(0xFFFFFFFFFFFFFFFF)


#wanted = 10*(10**6)
#a = set()
#while True:
#    to_sample = int(    (     (wanted-len(a)) * numpy.log(wanted-len(a))            )*1.1 )
#    print "Need", wanted-len(a), "generating", to_sample
#    a = a | set(numpy.random.random_integers( 0, num_examples-1, to_sample ))
#    if len(set(a))>=wanted:
#        a = list(a)[:wanted]
#        break
#    print len(a)
#print len(a)


#gc.collect()

#while True:
#  print len(a)

#import time
#t = time.time()
#batchsize = 10000
#batch = numpy.zeros( (batchsize,400) )
#for i in range(batchsize):
#    start = numpy.random.random_integers(0, len(train_wav)-400)
#    batch[i,:] = train_wav[start:start+400]
#    print i
#print time.time()-t
