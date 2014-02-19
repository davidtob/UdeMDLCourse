for i in {1..30}
do
   echo $i
   php learn_own_data_samples_mlp.php $i > "model_spec_$i.yaml"
done
