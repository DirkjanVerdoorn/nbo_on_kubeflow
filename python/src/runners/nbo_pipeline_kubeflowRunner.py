from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import absl
import yaml
from typing import Dict, List, Text, Optional

from tfx import v1 as tfx
import tensorflow_model_analysis as tfma
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
from src.utils.helper_utils import get_config

# Get configurations from yaml file
configs = get_config()

# Define AI platform training arguments
AI_PLATFORM_TRAINING_ARGS = {
    'project': configs['gc_project']['project_id'],
    'region': configs['gc_project']['gcp_region'],
    'masterConfig':{
        'imageUri': configs['pipeline']['pipeline_image']
    }
}

# Define AI platform serving arguments
AI_PLATFORM_SERVING_ARGS = {
    'model_name': configs['pipeline']['pipeline_name'].replace('-', '_'),
    'project_id': configs['gc_project']['project_id'],
    'regions': configs['gc_project']['gcp_region'],
    'machine_type': 'mls1-c1-m2'
}

# Define BEAM pipeline processing arguments
BEAM_PIPELINE_ARGS_BY_RUNNER = {
    'DataflowRunner':[
        '--runner=DataflowRunner',
        '--project=' + configs['gc_project']['project_id'],
        '--temp_location=' + os.path.join(configs['pipeline']['pipeline_root'], 'temp'),
        '--setup_file='+configs['pipeline']['setup_file'],
        '--region=' + configs['gc_project']['gcp_region'],
        '--experiment=upload_graph',
        '--experiment=use_runner_v2'
        '--machine_type=e2-standard-8',
        '--autoscaling_algorithm=THROUGHPUT_BASED',
        '--max_num_workers=10'
    ],
    'DirectRunner': [
        '--direct_running_mode=multi_processing',
        '--direct_num_workers=0' #-> will auto detect num workers needed
    ]
}

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    model_module_file: Text,
    transform_module_file: Text,
    target_name: Text,
    training_steps: int,
    validation_steps: int,
    beam_pipeline_args: List[Text],
    ai_platform_training_args: Optional[Dict[Text, Text]]=None,
    ai_platform_serving_args: Optional[Dict[Text, Text]]=None,
    enable_cache: Optional[bool]=False,
    enable_tuning: Optional[bool]=False,
    tune_train_steps: Optional[int]=None,
    tune_eval_steps: Optional[int]=None,
    parallel_trials: Optional[int]=None,
    import_hyperparameters: Optional[bool]=False,
    hyperparameters_path: Optional[Text]=None,
    enable_ai_platform_training: Optional[bool]=False,
    enable_ai_platform_serving: Optional[bool]=False,
    serving_model_dir: Optional[Text] = None
) -> tfx.dsl.Pipeline:
    """
    Description
    -----------
    Generates Tensorflow pipeline to deploy on Kubeflow for end to end machine learning
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    """
    
   # Create training arguments
    train_args = tfx.dsl.RuntimeParameter(
        name='train-args',
        default='{"num_steps": %s}' % training_steps,
        ptype=Text,
    )
    
    # Create evaluation arguments
    eval_args = tfx.dsl.RuntimeParameter(
        name='eval-args',
        default='{"num_steps": %s}' % validation_steps,
        ptype=Text,
    )
    
    # Initiate components list
    components = []
    
    # Generate output configuration for CsvExampleGen component
    output_config = tfx.proto.Output(
        split_config = example_gen_pb2.SplitConfig(splits=[
            # 70% of the data is used for training
            tfx.proto.SplitConfig.Split(name='train', hash_buckets=7),
            # 20% of the data is used for evaluation
            tfx.proto.SplitConfig.Split(name='eval', hash_buckets=2),
            # 10% of the data is ued for testing
            tfx.proto.SplitConfig.Split(name='test', hash_buckets=1)
        ],
        # Partition on specific feature
        partition_feature_name=configs['data']['partition_features'])
    )
    
    # The CsvExampleGen component brings data into the pipeline
    example_gen = tfx.components.CsvExampleGen(
        input_base=data_root, output_config=output_config
    )
    
    # Add example_gen component to component list
    components.append(example_gen)
    
    # The StatisticsGen component computes statistics over data for visualization
    # and example validation
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples']
    )
    
    # Add statistics_gen component to component list
    components.append(statistics_gen)
    
    # The SchemaGen generates a schema based on statistics files
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )
    
    # Add schema_gen component to component list
    components.append(schema_gen)
    
    # The ExampleValidator performs anomaly detection based on statistics and data schema
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'],
    )
    
    # Add example_validator component to component list
    components.append(example_validator)

    # The Transform performs transformations and feature engineering in training
    # and serving.
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file
    )

    # Add transform component to component list
    components.append(transform)

    # Performs hyperparamter tuning if enable_tuning=True
    if enable_tuning:
        if tune_train_steps == None or tune_eval_steps == None or parallel_trials == None:
            raise ValueError("""If tuning is enabled, both 'tune_train_steps',
                            'tune_eval_steps' and parallel_trials should be defined.""")
        
        if enable_ai_platform_training:

            # The tuner tunes the hyperparameters for model training to find the 
            # optimal set of hyperparamters
            tuner = tfx.extensions.google_cloud_ai_platform.Tuner(
                module_file=model_module_file,
                examples=transform.outputs['transformed_examples'],
                transform_graph=transform.outputs['transform_graph'],
                train_args=tfx.proto.TrainArgs(num_steps=tune_train_steps),
                eval_args=tfx.proto.EvalArgs(num_steps=tune_eval_steps),
                tune_args=tfx.proto.TuneArgs(
                    num_parallel_trials=parallel_trials
                ),
                custom_config={
                    tfx.extensions.google_cloud_ai_platform.experimental.TUNING_ARGS_KEY:
                        ai_platform_training_args,
                    tfx.extensions.google_cloud_ai_platform.experimental.REMOTE_TRIALS_WORKING_DIR_KEY:
                        os.path.join(pipeline_root, 'trials')
                }
            )

            # Add tuner component to component list
            components.append(tuner)

            # Set hyperparamters based on output of tuner
            hyperparameters = tuner.outputs['best_hyperparameters']
        
        else:
            # The tuner tunes the hyperparameters for model training to find the 
            # optimal set of hyperparamters
            tuner = tfx.components.Tuner(
                examples=transform.outputs['transformed_examples'],
                transform_graph=transform.outputs['transform_graph'],
                module_file=model_module_file,
                train_args=tfx.proto.TrainArgs(num_steps=tune_train_steps),
                eval_args=tfx.proto.EvalArgs(num_steps=tune_eval_steps),
                tune_args=tfx.proto.TuneArgs(
                    num_parallel_trials=parallel_trials
                )
            )

            # Add tuner component to component list
            components.append(tuner)

    else:
        # Check if previous parameters should be imported
        if import_hyperparameters:
            if hyperparameters_path == None:
                raise ValueError("""Import_hyperparameters is set to true, however, 
                                    hyperparameters_path is not defined,""")
            hyperparameters = tfx.dsl.components.common.importer.Importer(
                source_uri=hyperparameters_path,
                articact_type=standard_artifacts.HyperParameters
            )
        else:
            # If tuner is not enabled, and results from previous 
            # run are not available, set hyperparameters to None
            hyperparameters = None
    
    if enable_ai_platform_training:
        # Check if all necessary arguments are provided for AI platform training
        if None in [ai_platform_serving_args, ai_platform_training_args]:
            raise ValueError("""If enable_ai_platform_training is set to true
                            both ai_platform_serving_args and ai_platform_training_args
                            should be provided.""")

        trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
            module_file=model_module_file,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            schema=schema_gen.outputs['schema'],
            hyperparameters=hyperparameters,
            train_args=tfx.proto.TrainArgs(num_steps=training_steps),
            eval_args=tfx.proto.EvalArgs(num_steps=validation_steps),
            custom_config={
                tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                    ai_platform_training_args
            }
        )
    
    else:
        trainer = tfx.components.Trainer(
            module_file=model_module_file,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            schema=schema_gen.outputs['schema'],
            hyperparameters=hyperparameters,
            train_args=train_args,
            eval_args=eval_args,
        )

    # Add trainer component to component list
    components.append(trainer)

    # The resolver component gets the latest blessed model for model validation
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)
        ).with_id('latest_blessed_model_resolver')

    # Add resolver component to component list
    components.append(model_resolver)

    # The Evaluator component uses TFMA to compute evaluation statistics over features
    # of a model and performs quality validation of a candidate model
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                # No signature and preprocessing function needed as incoming data
                # is already transformed.
                label_key=target_name+'_xf',
                prediction_key='model_output',
                model_type=tfma.constants.TF_KERAS
            )],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(
                    class_name='AUC',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10}
                        )
                    )
                ),
                tfma.MetricConfig(
                    class_name='FairnessIndicators',
                    config='{ "thresholds": [0.25, 0.5, 0.75] }'
                ),
                tfma.MetricConfig(
                    class_name='Precision',
                    config='{"top_k": 3}'),
                tfma.MetricConfig(
                    class_name='Recall',
                    config='{"top_k": 3}')]
            )
        ]
    )

    evaluator = tfx.components.Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config,
        example_splits=['test']
    )

    # Add evaluator component to component list
    components.append(evaluator)

    # The pusher components pushes the model to product to serve for predictions
    if enable_ai_platform_serving:
        pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
            model=trainer.outputs['model'],
            model_blessing=evaluator.outputs['blessing'],
            custom_config={
                tfx.extensions.google_cloud_ai_platform.experimental
                .PUSHER_SERVING_ARGS_KEY: ai_platform_serving_args
            }
        )
    
    else:
        pusher = tfx.components.Pusher(
            model=trainer.outputs['model'],
            model_blessing=evaluator.outputs['blessing'],
            push_destination=tfx.proto.PushDestination(
                filesystem=tfx.proto.PushDestination.Filesystem(
                    base_directory=serving_model_dir
                )
            )
        )

    # Add pusher component to component list
    components.append(pusher)

    # Returns a tfx pipeline object
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        beam_pipeline_args=beam_pipeline_args
    )


def main():
    absl.logging.set_verbosity(absl.logging.INFO)
    
    metadata_config = tfx.orchestration.experimental.get_default_kubeflow_metadata_config()
    
    runner_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        tfx_image=configs['pipeline']['pipeline_image']
    )

    tfx.orchestration.experimental.KubeflowDagRunner(config=runner_config).run(
        create_pipeline(
            beam_pipeline_args=BEAM_PIPELINE_ARGS_BY_RUNNER['DataflowRunner'],
            ai_platform_training_args=AI_PLATFORM_TRAINING_ARGS,
            ai_platform_serving_args=AI_PLATFORM_SERVING_ARGS,
            enable_cache=True,
            enable_tuning=False,
            enable_ai_platform_training=True,
            enable_ai_platform_serving=True,
            **configs['pipeline'],
            **configs['data'],
            **configs['module_files'],
            **configs['training_and_tuning']
            **configs['gc_project']
        )
    )

if __name__ == '__main__':
    main()