import numpy
import cPickle

timit_location = "../timit/"

phonemes = numpy.array(cPickle.load(open(timit_location + 'readable/phonemes.pkl')))
words = numpy.array(cPickle.load(open(timit_location + 'readable/words.pkl')))
# Sound data
train_x_raw = numpy.load(timit_location + 'readable/train_x_raw.npy')
valid_x_raw = numpy.load(timit_location + 'readable/valid_x_raw.npy')
test_x_raw = numpy.load(timit_location + 'readable/test_x_raw.npy')
# list of every utterance of phoneme
train_phn = numpy.load(timit_location + 'readable/train_phn.npy')
valid_phn = numpy.load(timit_location + 'readable/valid_phn.npy')
test_phn = numpy.load(timit_location + 'readable/test_phn.npy')

train_seq_to_phn = numpy.load(timit_location + 'readable/train_seq_to_phn.npy')
valid_seq_to_phn = numpy.load(timit_location + 'readable/valid_seq_to_phn.npy')
test_seq_to_phn = numpy.load(timit_location + 'readable/test_seq_to_phn.npy')

train_seq_to_wrd = numpy.load(timit_location + 'readable/train_seq_to_wrd.npy')
valid_seq_to_wrd = numpy.load(timit_location + 'readable/valid_seq_to_wrd.npy')
test_seq_to_wrd = numpy.load(timit_location + 'readable/test_seq_to_wrd.npy') 

train_wrd = numpy.load(timit_location + 'readable/train_wrd.npy')
valid_wrd = numpy.load(timit_location + 'readable/valid_wrd.npy')
test_wrd = numpy.load(timit_location + 'readable/valid_wrd.npy')

train_spkr = numpy.load(timit_location + 'readable/train_spkr.npy')
spkrinfo = numpy.load(timit_location + 'readable/spkrinfo.npy').tolist().toarray()

def train_sentence_idx_to_wave(idx):
    return train_x_raw[idx]

def train_sentence_idx_to_word_idcs(idx):
    first_word_idx, last_word_idx = train_seq_to_wrd[idx]
    return range(first_word_idx,last_word_idx)

def valid_sentence_idx_to_word_idcs(idx):
    first_word_idx, last_word_idx = valid_seq_to_wrd[idx]
    return range(first_word_idx,last_word_idx)

def test_sentence_idx_to_word_idcs(idx):
    first_word_idx, last_word_idx = test_seq_to_wrd[idx]
    return range(first_word_idx,last_word_idx)

def train_sentence_idx_to_word_nums(idx):
    return train_wrd[train_sentence_idx_to_word_idcs(idx)][:,2]

def valid_sentence_idx_to_word_nums(idx):
    return valid_wrd[valid_sentence_idx_to_word_idcs(idx)][:,2]

def test_sentence_idx_to_word_nums(idx):
    return test_wrd[test_sentence_idx_to_word_idcs(idx)][:,2]

def word_num_to_word_str(idx):
    return words[idx]

def train_sentence_idx_to_words(idx):
    return word_num_to_word_str( train_sentence_idx_to_word_nums(idx) )

def train_sentence_idx_to_phoneme_idcs(idx):
    first_phoneme_idx, last_phoneme_idx = train_seq_to_phn[idx]
    return range( first_phoneme_idx, last_phoneme_idx)

def train_sentence_idx_to_phoneme_nums(idx):
    return train_phn[train_sentence_idx_to_phoneme_idcs(idx)][:,2]

def train_sentence_idx_to_phoneme_strs(idx):
    return phonemes[ train_sentence_to_phoneme_nums(idx) ]

def train_phoneme_idx_to_phoneme_num(idx):
    return train_phn[idx][2]

def train_phoneme_idx_to_phoneme_str(idx):
    return phonemes[ train_phoneme_idx_to_phoneme_num(idx) ]

def train_phoneme_idx_to_sentence_idx(idx):
    return find( map( lambda x: x[0]<=idx<x[1], train_seq_to_phn ) )[0]

def train_phoneme_idx_to_offsets(idx): # Start and end in sentence
    return train_phn[idx][0:2]

def train_phoneme_idx_to_wave(idx):
    sent_wave = train_sentence_idx_to_wave( train_phoneme_idx_to_sentence_idx(idx) )
    start, end = train_phoneme_idx_to_offsets(idx)
    return sent_wave[start:end]

def train_word_idx_to_offsets(idx):
    return train_wrd[idx][0:2]

def train_word_idx_to_word_num(idx):
    return train_wrd[idx][2]
    
def train_sentence_idx_to_word_idcs(idx):
    first_word_idx, last_word_idx = train_seq_to_wrd[idx]
    return range( first_word_idx, last_word_idx)