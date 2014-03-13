for speaker_id in 104 144 169 179
do
  for i in 2 9 13 14 16 21 22 28 29
  do
     echo $speaker_id $i
     for trainrate in 3 1 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
     do
       ~/NOBACKUP_HOME/bin/php learn_one_phone.php $i $speaker_id $trainrate > "$speaker_id/model_spec_$i-$trainrate.yaml"
     done 
  done
done
