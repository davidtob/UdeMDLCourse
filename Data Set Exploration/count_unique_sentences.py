# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:55:35 2014

@author: david
"""

execfile('load_data.py')

print "** Training set **"
print "Number of spoken sentences", len(train_x_raw)
largest_phn_idx = train_sentence_idx_to_phoneme_idcs(len(train_x_raw)-1)[-1]-1
print "Number of spoken phonemes", train_sentence_idx_to_phoneme_idcs(len(train_x_raw)-1)[-1]+1
print "Number of speakrers", len(unique(train_spkr))
print "List of all", len(phonemes), "phonemes", phonemes

print "Number of occurences of each phoneme"
phoneme_num_utterances = []
for i in range(len(phonemes)):
  num_utterances = sum(train_phn[:largest_phn_idx,2]==i)
  phoneme_num_utterances.append(num_utterances)
  print i, phonemes[i], '\t', num_utterances
  #print "<tr><td>",i,"</td><td>",phonemes[i], '</td><td>', sum(train_phn[:largest_phn_idx,2]==i),"</tr>"

print "Number of recordings of each sentence:"

def dict_count_add( dic, key ):
    if key in dic:
        dic[key] = dic[key] + 1
    else:
        dic[key] = 1

# Number of unique senences
ct = {}
for i in range(len(train_x_raw)):
    s = tuple(train_sentence_idx_to_word_nums(i))
    dict_count_add( ct, s )
for a in range(1,max(ct.values())+1):
    if sum(array(ct.values())==a)!=0:
        print "There are",sum(array(ct.values())==a), "sentences with",a,"recordings"

print "Total number of unique sentences:", len(ct.values())

sorted_ct = sorted(ct.iteritems(), key=operator.itemgetter(1),reverse=True)
print "Two most comment sentences:"
print " ".join( word_num_to_word_str(list(sorted_ct[0][0])) )
print " ".join( word_num_to_word_str(list(sorted_ct[1][0])) )

print "Number of phonetic variations of 10 most common sentences"
# Count phonetic variations of each sentence
for sent_wrds,count in sorted_ct[0:10]:
    ct_phn = {}
    for idx in range(len(train_x_raw)):
        if tuple(train_sentence_idx_to_word_nums(idx))==sent_wrds:
            phns = tuple(train_sentence_idx_to_phoneme_nums(idx))
            dict_count_add( ct_phn, phns )
    print len(ct_phn.values()),"unique phonetic variations out of",ct[sent_wrds],"recordings"

print "Some phonetic variations of most common sentences"
print "Sentence:", " ".join( word_num_to_word_str(list(sorted_ct[0][0])) )
to_show = 2;
for i in range(len(train_x_raw)):
    if tuple(train_sentence_idx_to_word_nums(i))==sorted_ct[0][0]:
        print " ".join( train_sentence_idx_to_phoneme_strs(i) )
        to_show -= 1
        if to_show==0: break