import numpy as np
import numpy
import cPickle
import gc
import os

data_path = os.environ["PYLEARN2_DATA_PATH"]


train_wav_unconcat = np.load(data_path + "/timit/readable/train_x_raw.npy")
train_lengths = map(lambda x:x.shape[0], train_wav_unconcat)
train_words_unconcat = np.load(data_path + "/timit/readable/train_x_words.npy")
train_phones_unconcat = np.load(data_path + "/timit/readable/train_x_phones.npy")
train_phonemes_unconcat = np.load(data_path + "/timit/readable/train_x_phonemes.npy")
f = open(data_path + "/timit/readable/train_path.pkl", "r")
train_paths = cPickle.load(f)
f.close()

valid_wav_unconcat = np.load(data_path + "/timit/readable/valid_x_raw.npy")
valid_lengths = map(lambda x:x.shape[0], valid_wav_unconcat)
valid_words_unconcat = np.load(data_path + "/timit/readable/valid_x_words.npy")
valid_phones_unconcat = np.load(data_path + "/timit/readable/valid_x_phones.npy")
valid_phonemes_unconcat = np.load(data_path + "/timit/readable/valid_x_phonemes.npy")
f = open(data_path + "/timit/readable/valid_path.pkl", "r")
valid_paths = cPickle.load(f)
f.close()

real_train_lengths = [0] + train_lengths + valid_lengths
real_train_intervals = np.cumsum(real_train_lengths)
real_train_wav_concat = np.zeros((real_train_intervals[-1]), dtype=numpy.uint16)
real_train_words_concat = np.zeros((real_train_intervals[-1]), dtype=numpy.uint16)
real_train_phones_concat = np.zeros((real_train_intervals[-1]), dtype=numpy.uint8)
real_train_phonemes_concat = np.zeros((real_train_intervals[-1]), dtype=numpy.uint8)
real_train_paths = train_paths + valid_paths

del train_paths
del valid_paths
gc.collect()

for i in range(len(train_wav_unconcat)):
    real_train_wav_concat[real_train_intervals[i]:real_train_intervals[i+1]] = train_wav_unconcat[i]
    real_train_words_concat[real_train_intervals[i]:real_train_intervals[i+1]] = train_words_unconcat[i]
    real_train_phones_concat[real_train_intervals[i]:real_train_intervals[i+1]] = train_phones_unconcat[i]
    real_train_phonemes_concat[real_train_intervals[i]:real_train_intervals[i+1]] = train_phonemes_unconcat[i]

del train_wav_unconcat
del train_words_unconcat
del train_phones_unconcat
del train_phonemes_unconcat
gc.collect()

for i in range(len(valid_wav_unconcat)):
    real_train_wav_concat[real_train_intervals[len(train_lengths)+i]:\
                    real_train_intervals[len(train_lengths)+i+1]] = \
                    valid_wav_unconcat[i]
    real_train_words_concat[real_train_intervals[len(train_lengths)+i]:\
                    real_train_intervals[len(train_lengths)+i+1]] = \
                    valid_words_unconcat[i]
    real_train_phones_concat[real_train_intervals[len(train_lengths)+i]:\
                    real_train_intervals[len(train_lengths)+i+1]] = \
                    valid_phones_unconcat[i]
    real_train_phonemes_concat[real_train_intervals[len(train_lengths)+i]:\
                    real_train_intervals[len(train_lengths)+i+1]] = \
                    valid_phonemes_unconcat[i]

del valid_wav_unconcat
del valid_words_unconcat
del valid_phones_unconcat
del valid_phonemes_unconcat

del train_lengths
del valid_lengths
gc.collect()

np.save("timit_concat/train_wav.npy", real_train_wav_concat)
np.save("timit_concat/train_words.npy", real_train_words_concat)
np.save("timit_concat/train_phones.npy", real_train_phones_concat)
np.save("timit_concat/train_phonemes.npy", real_train_phonemes_concat)
np.save("timit_concat/train_intervals.npy", real_train_intervals)
f = open("timit_concat/train_paths.pkl", "w")
cPickle.dump(real_train_paths, f)
f.close()

del real_train_wav_concat
del real_train_words_concat
del real_train_phones_concat
del real_train_phonemes_concat
del real_train_paths
del real_train_intervals
gc.collect()

test_wav_unconcat = np.load(data_path + "/timit/readable/test_x_raw.npy")
test_lengths = map(lambda x:x.shape[0], test_wav_unconcat)
test_words_unconcat = np.load(data_path + "/timit/readable/test_x_words.npy")
test_phones_unconcat = np.load(data_path + "/timit/readable/test_x_phones.npy")
test_phonemes_unconcat = np.load(data_path + "/timit/readable/test_x_phonemes.npy")
f = open(data_path + "/timit/readable/test_path.pkl", "r")
test_paths = cPickle.load(f)
f.close()

test_lengths = [0] + test_lengths
test_intervals = np.cumsum(test_lengths)

test_wav_concat = np.zeros((test_intervals[-1]))
test_words_concat = np.zeros((test_intervals[-1]))
test_phones_concat = np.zeros((test_intervals[-1]))
test_phonemes_concat = np.zeros((test_intervals[-1]))
f = open(data_path + "/timit/readable/test_path.pkl", "r")
test_paths = cPickle.load(f)
f.close()

for i in range(len(test_wav_unconcat)):
    test_wav_concat[test_intervals[i]:test_intervals[i+1]] = test_wav_unconcat[i]
    test_words_concat[test_intervals[i]:test_intervals[i+1]] = test_words_unconcat[i]
    test_phones_concat[test_intervals[i]:test_intervals[i+1]] = test_phones_unconcat[i]
    test_phonemes_concat[test_intervals[i]:test_intervals[i+1]] = test_phonemes_unconcat[i]

np.save("timit_concat/test_wav.npy", test_wav_concat)
np.save("timit_concat/test_words.npy", test_words_concat)
np.save("timit_concat/test_phones.npy", test_phones_concat)
np.save("timit_concat/test_phonemes.npy", test_phonemes_concat)
np.save("timit_concat/test_intervals.npy", test_intervals)
f = open("timit_concat/test_paths.pkl", "w")
cPickle.dump(test_paths, f)
f.close()

