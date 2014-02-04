import numpy
import cPickle

# This should point to the contents of the directory /data/lisa/data/timit/
# from the lab network.
timit_readable_location = "../timit/readable"

phonemes = numpy.array(cPickle.load(open(timit_readable_location + '/phonemes.pkl')))
words = numpy.array(cPickle.load(open(timit_readable_location + '/words.pkl')))
spkrinfo = numpy.load(timit_readable_location + '/spkrinfo.npy').tolist().toarray()
phone_map = cPickle.load(open(timit_readable_location + '/phone_map.pkl'))

def word_num_to_word_str(idx):
	return words[idx]


class TimitDataSet:
	def __init__(self,path_prefix):
		# Sound data
		self.x_raw = numpy.load(path_prefix + '_x_raw.npy')
		# list of every utterance of phoneme
		self.phn = numpy.load(path_prefix + '_phn.npy')
		# Which phonemes appear in each recorded sentence	
		self.seq_to_phn = numpy.load(path_prefix+ '_seq_to_phn.npy')
		# Which words appear in each recorded sentence
		self.seq_to_wrd = numpy.load(path_prefix + '_seq_to_wrd.npy')
		# Information on each word that appears
		self.wrd = numpy.load(path_prefix + '_wrd.npy')
		# Which speaker recorded each sentence
		self.spkr = numpy.load(path_prefix + '_spkr.npy')
	
	def number_of_recorded_sentences(self):
		return len(self.x_raw)
	
	def number_of_recorded_phonemes(self):
		return len(self.phn)
	
	def number_of_recorded_words(self):
		return len(self.wrd)
	
	def number_of_distinct_speakers(self):
		return len(self.spkr)

	def sentence_idx_to_wave(self,idx):
		return self.x_raw[idx]

	def sentence_idx_to_word_idcs(self,idx):
		first_word_idx, last_word_idx = self.seq_to_wrd[idx]
		return range(first_word_idx,last_word_idx)

	def sentence_idx_to_word_nums(self,idx):
		return self.wrd[self.sentence_idx_to_word_idcs(idx)][:,2]

	def sentence_idx_to_words(self,idx):
		return word_num_to_word_str( self.sentence_idx_to_word_nums(idx) )

	# For lack of a better terminology I call a "phoneme index"
	# the index of the a recoding of an utterance of a phoneme
	# in the train_phn array. I call the number that identifies
	# a specific phoneme (e.g. 'h#' or 'eng') the "phoneme number".
	def sentence_idx_to_phoneme_idcs(self,idx):
		first_phoneme_idx, last_phoneme_idx = self.seq_to_phn[idx]
		return range( first_phoneme_idx, last_phoneme_idx)

	def sentence_idx_to_phoneme_nums(self,idx):
		return self.phn[self.sentence_idx_to_phoneme_idcs(idx)][:,2]

	def sentence_idx_to_phoneme_strs(self,idx):
		return phonemes[ self.sentence_idx_to_phoneme_nums(idx) ]

	def phoneme_idx_to_phoneme_num(self,idx):
		return self.phn[idx][2]

	def phoneme_idx_to_phoneme_str(self,idx):
		return phonemes[ self.phoneme_idx_to_phoneme_num(idx) ]

	def phoneme_idx_to_sentence_idx(self,idx): # In which setence does this recording of a phoneme occur
		return find( map( lambda x: x[0]<=idx<x[1], self.seq_to_phn ) )[0]

	def phoneme_idx_to_offsets(self,idx): # Start and end in sentence
		return self.phn[idx][0:2]

	def phoneme_idx_to_wave(self,idx):
		sent_wave = self.sentence_idx_to_wave( self.phoneme_idx_to_sentence_idx(idx) )
		start, end = self.phoneme_idx_to_offsets(idx)
		return sent_wave[start:end]

	def word_idx_to_offsets(self,idx):
		return self.wrd[idx][0:2]

	def word_idx_to_word_num(self,idx):
		return self_wrd[idx][2]
    
	def sentence_idx_to_word_idcs(self,idx):
		first_word_idx, last_word_idx = self.seq_to_wrd[idx]
		return range( first_word_idx, last_word_idx)

train = TimitDataSet(timit_readable_location + '/train')
valid = TimitDataSet(timit_readable_location + '/valid')
test = TimitDataSet(timit_readable_location + '/test')
