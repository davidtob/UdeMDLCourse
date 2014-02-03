# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:57:44 2014

@author: david
"""


import alsaaudio
import time
import matplotlib.pyplot as plt
from matplotlib import animation


device = alsaaudio.PCM()
device.setchannels(1)
device.setrate(16000)
device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
device.setperiodsize(1)

def play_and_pause( wave ):
    device.write( wave )
    pause( len(wave)/16000.0 )

def play_and_display_waveform( wave ):
    start_time = time.time()    
    device.write( wave )
    ax = fig2.add_subplot(111)
    ax.plot( numpy.arange(len(wave)), wave )

def animate(i,line,offsets,texts,start_time):
    t = time.time()-start_time
    line.set_data( [t*16000], [-10000,10000] )
    curr_phn = find(map( lambda x: x[0]<=t<x[1], offsets ))
    for text,offset in zip(texts,offsets):
        if offset[0]<=t*16000<offset[1]:
            text.set_color('r')
            newtext = text
        else:
            text.set_color('k')
    return tuple([line]+  texts)

def play_and_display_waveform_with_phonemes_and_words( sent_idx ):
    fig = plt.figure()
    fig.suptitle( " ".join(train_sentence_idx_to_words(sent_idx)))
    wave = train_sentence_idx_to_wave(sent_idx)
    phn_idcs = train_sentence_idx_to_phoneme_idcs(sent_idx)
    ax = fig.add_subplot(111)
    #ax.plot( numpy.arange(len(wave)), wave )
    line = ax.plot( wave )
    colors = ['b', 'g', 'r', 'c']
    offsets = []
    texts = []
    for phn_idx,i in zip(phn_idcs,range(len(phn_idcs))):
        start, end = train_phoneme_idx_to_offsets(phn_idx)
        offsets.append( (start,end) )
        ax.axvspan(start, end, alpha=.2, color=colors[i%len(colors)])
        trans = ax.get_xaxis_transform()
        text = ax.text((start + end) / 2, (i%3)/20.0+0.8, train_phoneme_idx_to_phoneme_str(phn_idx), transform=trans)
        texts.append(text)
            
    
    word_idcs = train_sentence_idx_to_word_idcs(sent_idx)
    for wrd_idx,i in zip(word_idcs,range(len(word_idcs))):
        start, end = train_word_idx_to_offsets(wrd_idx)
        word_str = train_word_num_to_word_str(train_word_idx_to_word_num(wrd_idx))
        offsets.append( (start,end) )
        text = ax.text((start + end) / 2, (i%3)*1000, word_str)
        texts.append(text)
    
    line, = ax.plot( [0,0], [-10000,10000])
    start_time = time.time()
    device.write(wave)
    anim = animation.FuncAnimation(fig, animate, fargs=(line,offsets,texts,start_time,),
                               frames=100, interval=100, blit=True, repeat=False)
    plt.show()

    pause(len(wave)/16000.0)
    
n = 3;
print "Showing and playing", n, "random sentences"

for i in numpy.random.choice( range(train_x_raw.shape[0]),n):
  print 'Sentence number:', i
  print " ".join( train_sentence_idx_to_words(i) ) # Print the words in the sentence
  print " ".join( train_sentence_to_phoneme_strs(i) ) # Print the phonemes in the sentence
  clf()
  plot( train_sentence_idx_to_wave(i) )
  play_and_display_waveform_with_phonemes_and_words( i )
  
n = 0;
m = 4;
print "Showing and playing",m,"random variations of ",n,"random phonemes"
#for i in numpy.random.choice( range(lsszen(train_phn)), n ):
for i in numpy.random.choice( range(train_seq_to_phn[-1][1]), n ):
    print "Phoneme:", train_phoneme_idx_to_phoneme_str(i)
    print " ".join( train_sentence_idx_to_words( train_phoneme_to_sentence_idx( i ) ) )
    clf()
    plot(train_phoneme_idx_to_wave(i) )
    print "Phoneme length: ", len(train_phoneme_idx_to_wave(i))/16000.0,"seconds"
    for j in range(5):
        play_and_pause( train_phoneme_idx_to_wave(i) )
        pause(1)