import numpy
import cPickle

timit_location = "../timit/"

train_x_raw = numpy.load(timit_location + 'readable/train_x_raw.npy');
train_phn = numpy.load(timit_location + 'readable/train_phn.npy')
train_seq_to_phn = numpy.load(timit_location + 'readable/train_seq_to_phn.npy')
train_seq_to_wrd = numpy.load(timit_location + 'readable/train_seq_to_wrd.npy') 
phonemes = cPickle.load(open(timit_location + 'readable/phonemes.pkl'))
words = cPickle.load(open(timit_location + 'readable/words.pkl'))
train_wrd = numpy.load(timit_location + 'readable/train_wrd.npy')
train_spkr = numpy.load(timit_location + 'readable/train_spkr.npy')
spkrinfo = numpy.load(timit_location + 'readable/spkrinfo.npy').tolist().toarray()

print "** Training set **"
print "Number of spoken sentences", len(train_x_raw)
print "Number of spoken phonemes", len(train_phn)
print "List of all", len(phonemes), "phonemes", phonemes

print "Number of occurences of each phoneme"
for i in range(len(phonemes)):
  print phonemes[i], '\t', sum(train_phn[:,2]==i)

print "Phonemes and words in first few sentences"
for i in range(10):
  first_phoneme, last_phoneme = train_seq_to_phn[i]
  first_word, last_word = train_seq_to_wrd[i]
  print '#', i
  for idx in range(first_phoneme, last_phoneme):
    print phonemes[train_phn[idx,2]],
  print 
  for idx in range(first_word, last_word):
    print words[train_wrd[idx,2]], 
  print
