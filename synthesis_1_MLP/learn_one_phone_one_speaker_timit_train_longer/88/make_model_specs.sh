for i in {0..29}
do
   echo $i
   php learn_one_phone.php $i > "model_spec_$i.yaml"
done
