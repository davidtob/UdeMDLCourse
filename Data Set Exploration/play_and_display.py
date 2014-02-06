# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:57:44 2014

@author: david
"""


import scipy
import scipy.io
import scipy.io.wavfile
import matplotlib.pyplot as plt
from matplotlib import animation

def offset_to_ms( off ):
    return off/16000.0*1000.0


def add_phones_to_plot(ax, sent_idx):
    # Plot phones and record positions for animation
    phn_idcs = train.sentence_idx_to_phoneme_idcs(sent_idx)
    colors = ['b', 'g', 'r', 'c']
    offsets = []
    texts = []
    for phn_idx,i in zip(phn_idcs,range(len(phn_idcs))):
        start, end = train.phoneme_idx_to_offsets(phn_idx)
        start = offset_to_ms(start); end = offset_to_ms(end)
        offsets.append( (start,end) )
        ax.axvspan(start, end, alpha=.2, color=colors[i%len(colors)])
        text = ax.text((start + end) / 2, (i%3)/4.0+1, train.phoneme_idx_to_phoneme_str(phn_idx), horizontalalignment="center")
        texts.append(text)
    return texts, offsets

def add_words_to_plot(ax,sent_idx):
    offsets = []
    texts = []
    word_idcs = train.sentence_idx_to_word_idcs(sent_idx)
    # Plot words and record positions for animation
    for wrd_idx,i in zip(word_idcs,range(len(word_idcs))):
        start, end = train.word_idx_to_offsets(wrd_idx)
        start = offset_to_ms(start); end = offset_to_ms(end)
        word_str = word_num_to_word_str(train.word_idx_to_word_num(wrd_idx))
        offsets.append( (start,end) )
        text = ax.text((start + end) / 2, 2+(i%2)/5.0, word_str, horizontalalignment="center")
        texts.append(text)

        y = 1.8 + (i%2)*0.1
        ax.annotate(
            '', xy=(start, y), xycoords = 'data',
            xytext = (end, y), textcoords = 'data',
            arrowprops = {'arrowstyle':'|-|'} )#, transform = trans)
    return texts,offsets


def plot_waveform_with_phones_and_words( sent_idx ):
    # Get sentence data
    wave = train.sentence_idx_to_wave(sent_idx)
    wave = wave/float(max( max(wave), -min(wave) )) # normalize
    clip_length = offset_to_ms(len(wave)) # length in milliseconds

    # Plot waveform
    fig = plt.figure()
    fig.suptitle( " ".join(train.sentence_idx_to_words(sent_idx)))
    ax = fig.add_subplot(111)
    ax.plot( numpy.linspace(0, clip_length, len(wave)), wave )

    # Create the line showing the current position
    vertline, = ax.plot( [0,0], [-1,3])

    #trans = ax.get_xaxis_transform()
    phn_texts, phn_offsets = add_phones_to_plot(ax, sent_idx)
    wrd_texts, wrd_offsets = add_words_to_plot(ax, sent_idx)

    return (fig, vertline, phn_texts+wrd_texts,phn_offsets+wrd_offsets)

def plot_image_with_phones_and_words( img, sent_idx ):
    # Get sentence data
    wave = train.sentence_idx_to_wave(sent_idx)
    wave = wave/float(max( max(wave), -min(wave) )) # normalize
    clip_length = offset_to_ms(len(wave)) # length in milliseconds

    # Plot waveform
    fig = plt.figure()
    fig.suptitle( " ".join(train.sentence_idx_to_words(sent_idx)))
    ax = fig.add_subplot(111)
    im = plt.imshow( img, extent=(0, clip_length, -1,0.9),aspect = clip_length/2/2 )#numpy.linspace(0, clip_length, len(wave)), wave )
    im.set_cmap('hot')    
    
    plt.axis( (0,clip_length,-1,3))

    # Create the line showing the current position
    vertline, = ax.plot( [0,0], [-1,3])

    #trans = ax.get_xaxis_transform()
    phn_texts, phn_offsets = add_phones_to_plot(ax, sent_idx)
    wrd_texts, wrd_offsets = add_words_to_plot(ax, sent_idx)

    return (fig, vertline, phn_texts+wrd_texts,phn_offsets+wrd_offsets)

def animate(i,vertline,offsets,texts,dt):
    t = i*dt # dt is number of milliseconds per frame
    vertline.set_data( [t], [-1,3] )
    for text,offset in zip(texts,offsets):
        if offset[0]<=t<offset[1]:
            text.set_color('r')
            newtext = text
        else:
            text.set_color('k')
    return tuple([vertline]+  texts)

def animate_waveform_with_phones_and_words( filename, sent_idx ):
    fig,vertline,texts,offsets = plot_waveform_with_phones_and_words( sent_idx )    
    
    # Set up the animation
    fps = 20
    interval = 1000.0/fps # Time that passes per frame
    num_frames = int(numpy.ceil(len( train.sentence_idx_to_wave( sent_idx ) )/16000.0*20.0))
    # had to install libavcodec-extra-53 for the below to work, see http://matplotlib.1069221.n5.nabble.com/Saving-animations-td39234.html
    anim = animation.FuncAnimation(fig, animate, fargs=(vertline,offsets,texts,interval),
                               frames=num_frames, interval=interval, blit=True, repeat=False)
    print "Saving animation"
    anim.save(filename, fps=fps, codec= 'libx264' ) # this can be quite slow

import subprocess
def merge_files( in1, in2, out ):
    # Needs ffmpeg to be installed
    print in1, in2, out
    cmd = ("ffmpeg -y -i " + in1 + " -i " + in2 + " -vcodec libx264 -vpre medium " + out).split(" ")
    proc = subprocess.Popen(cmd)
    proc.wait()

import os
def delete_temp_files():
	try:
	    os.remove('.play_and_display_temp.mp4')
	except:
	    pass
	try:
	    os.remove('.play_and_display_temp.wav')
	except:
	    pass

def make_video( dataset, sent_idx, output_fn ):
	delete_temp_files()
	print "Animating"
	animate_waveform_with_phones_and_words( '.play_and_display_temp.mp4', sent_idx )
	print "Saving sound"
	scipy.io.wavfile.write(".play_and_display_temp.wav", 16000, train.sentence_idx_to_wave(sent_idx) )
	print "Merging"
	merge_files( '.play_and_display_temp.mp4', '.play_and_display_temp.wav', output_fn)
	#delete_temp_files()


from base64 import b64encode
def html_video( fn ):
	video = open(fn, "rb").read()
	video_encoded = b64encode(video).decode('ascii')
	video_tag = '<video controls alt="test" src="data:video/x-m4v;base64,{0}">'.format(video_encoded) + '</video>'
	return video_tag	

