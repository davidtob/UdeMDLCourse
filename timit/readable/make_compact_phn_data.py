import numpy

# Contains phn indices for each sentence
train_seq_to_phn = numpy.load(open("train_seq_to_phn.npy"))

# Contains mapping of phn index to offset and phn number
train_phn = numpy.load(open("train_phn.npy"))

train_x_compact_phone_nums    = numpy.zeros( len(train_seq_to_phn), dtype=object )
train_x_compact_phone_offsets = numpy.zeros( len(train_seq_to_phn), dtype=object )
for i,seq in enumerate(train_seq_to_phn):
  first_phn = seq[0]
  last_phn  = seq[1]
  phn_numbers     = numpy.zeros( last_phn - first_phn, dtype=numpy.uint8 )
  phn_transitions = numpy.zeros( last_phn - first_phn, dtype=numpy.uint32 )
  for j in range(first_phn,last_phn):
    phn_numbers[j-first_phn]     = train_phn[j,2]
    phn_transitions[j-first_phn] = train_phn[j,0]
  train_x_compact_phone_nums[i]    = phn_numbers
  train_x_compact_phone_offsets[i] = phn_transitions

print train_x_compact_phone_nums[0]
print train_x_compact_phone_offsets[0]

numpy.save( "train_x_compact_phone_nums.npy", train_x_compact_phone_nums )
numpy.save( "train_x_compact_phone_offsets.npy", train_x_compact_phone_offsets )
