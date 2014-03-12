<?php 
function choice( $array ) {
	return $array[ array_rand( $array ) ];
}
function p( $x )
{
	echo $x;
	return $x;
}

$seed = $argv[1];
srand( $seed );
?>
!obj:pylearn2.train.Train {
    dataset: &train !obj:timit_data.TimitPredFramesForPhn {
        xsamples: <?php $xsamples = p( choice( [ 140, 160, 180, 200, 220, 240, 260 ] ) ); ?>,
        ysamples: <?php $ysamples = p( choice( [ 1,2,3,4,5] ) ); ?>,
        rescale: <?php $rescale = p( choice( ['framemax', 'phonemax', 'sentmax', 'datasetmax'] ) ); ?>,
        speaker_info: <?php $speaker_info = p( choice( ['False'] ) ); ?>,
        speaker_id: <?php $speaker_id = p( 104 );?>,
        phone: <?php $phone = p( 'aa' )?>,
        num_examples: <?php $num_examples = p( 100000 ); ?>,
        trainorvalid: 'train',
        train_valid_split: <?php $train_valid_split = p( 0.8 ); ?>,
        seed: <?php $data_seed = p( 0 );?>,
    },
    model: !obj:pylearn2.models.mlp.MLP {
<?php $num_inputs = $xsamples+($speaker_info=='True')*26; ?>
        layers: [
                <?php $num_hidden_layers = choice( [1,2,3,4,5] );
                for( $i=1 ; $i<= $num_hidden_layers ; $i++ )
                {
				 ?>!obj:pylearn2.models.mlp.RectifiedLinear {
                     layer_name: 'h<?php p( $i );?>',
                     dim: <?php p( $num_inputs ); ?>,
                     irange: <?php p( sqrt( 6.0/( 2*$num_inputs ) ) ); ?>
                   },
                 <?php }
                 ?>!obj:pylearn2.models.mlp.Linear {
                     layer_name: 'y',
                     dim: <?php p( $ysamples ); ?>, 
                     irange: <?php p( sqrt( 6.0/($ysamples+$num_inputs) ) ); ?>,
                 },
                ],
        nvis: <?php p( $num_inputs ); ?>,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: <?php $learning_rate = p( choice( [.0001, 0.00001, 0.000001 ] ) )?>,
        init_momentum: .5,
        monitoring_dataset: {
            'train': *train,
            'valid': !obj:timit_data.TimitPredFramesForPhn {
                xsamples: <?php p( $xsamples ); ?>,
                ysamples: <?php p( $ysamples ); ?>,
                rescale: <?php p( $rescale ); ?>,
                speaker_info: <?php p( $speaker_info ); ?>,
                speaker_id: <?php p( $speaker_id );?>,
                phone: <?php p( $phone ); ?>,
                num_examples: <?php p( $num_examples ) ?>,
                trainorvalid: 'valid',
                train_valid_split: <?php p( $train_valid_split );?>,
                seed: <?php p( $data_seed );?>,
            }
        },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
                !obj:pylearn2.costs.mlp.Default {
                }, !obj:pylearn2.costs.mlp.<?php $regularization = p( choice([ 'WeightDecay', 'L1WeightDecay' ]) );?> {
<?php $reg_const = choice( [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.000001, 0.000001, 0.000001, 0.0000001, '0.0' ] );?>
                        coeffs: [ <?php for( $i=1;$i<=$num_hidden_layers+1;$i++) { echo $reg_const . ","; } ?> ],
                }
                ]
        }, 
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 200
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
            channel_name: 'valid_objective',
            save_path: "best<?php echo $seed;?>.pkl",
        },
        !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: 1,
            saturate: 10,
            final_momentum: .99
        }
    ],
    save_path: "progress<?php echo $seed;?>.pkl",
    save_freq: 1
}

<?php
if( !file_exists( 'hparams.txt' ) )
{
	file_put_contents( 'hparams.txt',
	"seed \t xsamples \t ysamples \t rescale \t speaker_info \t num_hidden_layers\t learning_rate \t Regularization \t Regularization const \n");
}
file_put_contents( 'hparams.txt', $seed . "\t" . $xsamples . "\t" . $ysamples . "\t" . $rescale . "\t" . $speaker_info .
                                          "\t" . $num_hidden_layers . "\t" . $learning_rate . 
                                          "\t" . $regularization . "\t" . $reg_const . "\n", FILE_APPEND );
?>
