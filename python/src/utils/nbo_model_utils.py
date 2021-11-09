from datetime import datetime

import os
from typing import List, Text, Optional, Any

import pandas as pd
import kerastuner

from tensorflow.python.eager.context import context
from absl import logging

import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorflow_transform as tft
import tfx.v1 as tfx
from tfx_bsl.public import tfxio

from google.cloud import storage
from google.api_core import page_iterator

from src.utils.helper_utils import _transformed_name, get_config
from src.utils.helper_utils import get_vocabulary, get_tft_vocab

# Get configurations from yaml file
configs = get_config()

####################################
######## Vocabulary Configs ########
#################################### 

vocabulary, _vocab_size = get_vocabulary(
    vocab_root=configs['pipeline']['vocab_root_csv'],
    column='vocabulary'
)

# Get get vocabulary length of all vocabulary features
EMB_IN_DIMS = []
for key in configs['data']['vocab_feature_keys']:
    EMB_IN_DIMS.append(
        get_tft_vocab(
            name=_transformed_name(key),
            bucket=configs['gcs_buckets']['output_bucket'].replace('gs://', ''),
            directory=configs['pipeline']['raw_transform_output'])
    )

###################################
######### Model Functions #########
###################################

def weightedBinaryCrossentropy(true, pred, weight_zero=0.016, weight_one=1):
    """
    Description
    -----------
    Calculates the weighted binary cross-entropy loss.

    Arguments:
    ----------
    true (array): array containing true labels
    pred (array): array containing predicted labels
    weight_zero (float): weight for the non-active classes in label
    weight_one (float): weight for the active classes in label
    """

    binaryCross = tf.keras.backend.binary_crossentropy(true, pred)

    # Apply the weights
    weights = true * weight_one + (1. -true) * weight_zero
    weightedBinaryCross = weights * binaryCross

    return tf.keras.backend.mean(weightedBinaryCross)


@tf.keras.utils.register_keras_serializable()
class SimpleHierachicalAttention(tf.keras.layers.Layer):
    
    def __init__(self, units: Optional[int]=128, **kwargs):
        """
        Description
        -----------

        Parameters
        ----------
        units (int): number of dense units for generating final context layer (defaults to 128)

        Returns
        -------
        Tensor containing context vector with attention weights
        
        """

        self.units = units
        super().__init__(**kwargs)

    def __call__(self, inputs):
        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])

        # Calculate dense score based on inputs dot W matrix
        dense_score = tf.keras.layers.Dense(
            units=hidden_size,
            use_bias=False,
            name='dense_attention_score_vec'
        )(hidden_states)

        # Get last hidden state
        h_t = tf.keras.layers.Lambda(
            lambda x: x[:, -1, :],
            output_shape=(hidden_size,),
            name='last_hidden_state'
        )(hidden_states)

        # Get score based on last hidden state (h_t) dot dense score (dense_score)
        score = tf.keras.layers.Dot(
            axes=[1,2],
            name='attention_score'
        )([h_t, dense_score])

        # Get attention weights based on softmax activation on score value
        attention_weights = tf.keras.layers.Activation(
            activation='softmax',
            name='attention_weigths'
        )(score)

        # Get contex vector by dot product between hidden states and attetion weights
        context_vector = tf.keras.layers.Dot(
            axes=[1,1],
            name='context_vector'
        )([hidden_states, attention_weights])

        pre_activation = tf.keras.layers.Concatenate()(
            [context_vector, h_t]
        )

        attention_vector = tf.keras.layers.Dense(
            units=self.units,
            use_bias=False,
            activation='tanh',
            name='attention_vector'
        )(pre_activation)

        return attention_vector


@tf.keras.utils.register_keras_serializable()
class predictionsPostprocessing(tf.keras.layers.Layer):

    def __init__(self, labels:List[Text], top_k_conf:Optional[int]=3, **kwargs):
        self.labels = labels
        self.top_k = top_k_conf
        super(predictionsPostprocessing, self).__init__(**kwargs)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        tf_labels = tf.constant([self.labels], dtype='string')
        tf_labels = tf.tile(tf_labels, [batch_size, 1])

        top_k = tf.nn.top_k(x, k=self.top_k, sorted=True, name='top_k').indices

        top_k_confi = tf.gather(x, top_k, batch_dims=1)
        top_k_labels = tf.gather(tf_labels, top_k, batch_dims=1)
        
        return [top_k_confi, top_k_labels]

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        top_shape = (batch_size, self.top_k)
        return [top_shape, top_shape]

    def get_config(self):
        config = {'labels':self.labels, 'top_k_conf':self.top_k}
        base_config = super(predictionsPostprocessing, self).get_config()
        return dict(list(base_config.items)) + list(config.items())


def _input_fn(
            file_patter: List[Text],
            data_accessor: tfx.components.DataAccessor,
            tf_transform_output: tft.TFTransformOutput,
            batch_size: Optional[int]=configs['model_configs']['epochs']):

    return data_accessor.tf_dataset_factory(
        file_patter,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, 
            label_key=_transformed_name(configs['data']['target_name']),
            drop_final_batch=True,
        ),
        tf_transform_output.transformed_metadata.schema
    ).repeat()


###################################
######### Model Signatures ########
###################################

def _get_tf_examples_serving_signature(model, tf_transform_output):
    """
    Returns a serving signature that accepts `tensorflow.Example`.
    """

    # We need to track the layers in the model in order to save it.
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples_serving')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """
        Returns the output to be used in the serving signature.
        """

        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(configs['data']['target_name'])
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """
    Returns a serving signature that applies tf.Transform to features.
    """

    # We need to track the layers in the model in order to save it.
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples_transform')
    ])
    def transform_features_fn(serialized_tf_example):
        """
        Returns the transformed_features to be fed as input to evaluator.
        """
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def _get_xai_preprocess_fn(model, tf_transform_output):
    """
    In order to use the XAI we need to separate the preprocessing 
    from the actual computations.

    Args:
    serialized_tf_examples (tensor): a tensor containing a serialized example.

    Returns:
    type: Description of returned object.
    """

    model.tft_layer_xai = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples_xai')
    ])
    def xai_preprocess_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(configs['data']['target_name'])
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer_xai(parsed_features)
        logging.info('xai_transformed_features = %s', transformed_features)
        return transformed_features

    return xai_preprocess_fn


###################################
############## Model ##############
###################################

def _get_hyperparameters():
    """
    Returns hyperparameters for building Keras model.
    """

    hp = kerastuner.HyperParameters()

    # Define search space for hyperparamters
    hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4], default=configs['model_configs']['learning_rte'])
    hp.Float('lstm_dropout_1', min_value=0.1, max_value=0.3, default=0.25, step=0.5)
    hp.Float('lstm_dropout_2', min_value=0.1, max_value=0.3, default=0.25, step=0.5)
    hp.Choice('lstm_units_1', [32, 64, 128], default=32)
    hp.Choice('lstm_units_2', [16, 32, 64], default=16)
    hp.Choice('batch_size', [128, 256, 512], default=configs['model_configs']['batch_size'])

    return hp


def _build_keras_model(hparams:kerastuner.HyperParameters, **kwargs) -> tf.keras.Model:
    """
    Description
    -----------
    Generates a fusion network consisting of recurrent and dense architecture.
    Sequences of market baskets are processed through the recurrent architecture, whereas
    the customer characteristics are processed by the dense architecture.

    
    Returns
    -------

    """

    input_layers = []

    def build_characteristics_model(**kwargs):
        # Define empty helper lists
        charact_layers = []
        charact_emb_layers = []

        # Add all vocabulary features to character input layers
        for key in configs['data']['vocab_feature_keys']:
            charact_emb_layers.append(
                tf.keras.layers.Input(
                    shape=(1,),
                    name=_transformed_name(key),
                    ragged=False,
                    dtype=tf.int32
                )
            )

        # Add +1 to remove negative values before embedding
        charact_emb_add = [i+1 for i in charact_emb_layers]
        
        # Concatenate different embed charact layers
        charact_emb = tf.keras.layers.Concatenate(
                            axis=-1)(charact_emb_add)

        # Embed concatenated charact layers
        embedded_charact = tf.keras.layers.Embedding(
                input_dim=max(EMB_IN_DIMS)+1,
                output_dim=configs['model_configs']['charact_embedding_dim']
            )(charact_emb)

        # Add all integer features to character input layers and cast to float
        for key in configs['data']['dense_int_feature_keys']:
            charact_layers.append(
                    tf.keras.layers.Input(
                        shape=(1,),
                        name=_transformed_name(key),
                        ragged=False,
                        # is float as int features are converted to float during preprocessing
                        dtype=tf.float32 
                    )
                )

        # Add all float features to character input layers
        for key in configs['data']['dense_float_feature_keys']:
            charact_layers.append(
                tf.keras.layers.Input(
                    shape=(1,),
                    name=_transformed_name(key),
                    ragged=False,
                    dtype=tf.float32
                )
            )

        # Add all bool features to character input layers
        for key in configs['data']['bool_feature_keys']:
            charact_layers.append(
                tf.keras.layers.Input(
                    shape=(1,),
                    name=_transformed_name(key),
                    ragged=False,
                    dtype=tf.float32
                )
            )
        
        # Concatenate charact_layers
        charact_layers_con = tf.keras.layers.Concatenate(
            axis=-1)(charact_layers)

        # Run dense charact layers to dense network
        charact_layers_dense = tf.keras.layers.Dense(
            units=configs['model_configs']['charact_embedding_dim'],
            activation='relu'
            )(charact_layers_con)

        # Add extra dimension to dense charact layers before concatenation
        charact_layers_dense_expand = tf.expand_dims(
        input=charact_layers_dense,
            axis=1
        )

        # Combine embedded and dense charact inputs
        charact_output = tf.keras.layers.Concatenate(
            axis=1)([embedded_charact, charact_layers_dense_expand])

        charact_output_flatten = tf.keras.layers.Flatten()(charact_output)
        
        # Extend input layes list with charact input layers   
        input_layers.extend(charact_emb_layers)
        input_layers.extend(charact_layers)

        return charact_output_flatten

    # Get character processed inputs
    charact_output = build_characteristics_model()

    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=None,
        standardize=None,
        ngrams=None,
        output_sequence_length=configs['model_configs']['max_basket_len'],
        pad_to_max_tokens=True,
        vocabulary=vocabulary
    )

    string_layers = []
    for key in configs['data']['string_feature_keys']:
        string_layers.append(
            tf.keras.layers.Input(
                shape=(),
                name=_transformed_name(key),
                ragged=False,
                dtype=tf.string
            )
        )

    input_layers.extend(string_layers)

    # Run basket inputs through first LSTM layer
    lstm_out = tf.keras.layers.LSTM(
        hparams.get('lstm_units_1'),
        return_sequences=True,
        dropout=hparams.get('lstm_dropout_1'))(
            tf.cast(
                tf.keras.layers.Concatenate(axis=1)(
                    [
                        tf.expand_dims(vectorize_layer(i), axis=1)
                        for i in string_layers
                    ]
                ),
                tf.float32
            )
        )
    
    lstm_out = tf.keras.layers.LSTM(
        units=hparams.get('lstm_units_2'),
        return_sequences=True,
        dropout=hparams.get('lstm_dropout_2')
    )(lstm_out)

    # Get attention
    attention = SimpleHierachicalAttention(
        units=configs['model_configs']['attention_units']
    )(lstm_out)
    
    # Flatten attention weights
    attention_flatten = tf.keras.layers.Flatten()(attention)
    
    # Note that W[x,y]=W1x+W2y where [ ] denotes concat and W is split horizontally into W1 and W2. 
    # Compare this to W(x+y)=Wx+Wy. So you can interpret adding as a form of concatenation where 
    # the two halves of the weight matrix are constrained to W1=W2.

    # Make sure shape of dense charact values is equal to attention output
    charact_output_dense = tf.keras.layers.Dense(
        units=configs['model_configs']['attention_units'],
        activation='relu'
    )(charact_output)

    # Concatenate attention output with charact output
    attention_charact_con = tf.keras.layers.Concatenate(
        axis=-1
    )([attention_flatten, charact_output_dense])

    dense_1 = tf.keras.layers.Dense(
        units=4048,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L1(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)
    )(attention_charact_con)

    dropout_dense_1 = tf.keras.layers.Dropout(
        rate=0.20
    )(dense_1)

    dense_2 = tf.keras.layers.Dense(
        units=2048,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L1(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)
    )(dropout_dense_1)

    dropout_dense_2 = tf.keras.layers.Dropout(
        rate=0.1
    )(dense_2)

    # Concatenated output to dense layer
    dense_3 = tf.keras.layers.Dense(
        units=2048,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.L1(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)
    )(dropout_dense_2)

    # Normalize batch
    normalized_out = tf.keras.layers.BatchNormalization()(dense_3)

    dense_out = tf.keras.layers.Dense(
        units=1024,
        activation='relu'
    )(normalized_out)

    # Get predictions
    out = tf.keras.layers.Dense(
        units=_vocab_size,
        activation='sigmoid',
        name='model_output'
    )(dense_out)

    # Define the base model
    model = tf.keras.Model(
        inputs=input_layers,
        outputs=out
    )
    
    # Compile the base model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=hparams.get('learning_rate')),
        loss={
            'model_output': weightedBinaryCrossentropy
        },
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(top_k=5),
            tf.keras.metrics.Precision(top_k=10),
            tf.keras.metrics.Recall(top_k=5),
            tf.keras.metrics.Recall(top_k=10)
        ]
    )

    # Generate the base model summary
    model.summary(print_fn=logging.info)

    return model

def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
    """Build the tuner using the KerasTuner API.
    Args:
        fn_args: Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    Returns:
        A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                        model , e.g., the training and validation dataset. Required
                        args depend on the above tuner's implementation.
    """

    tuner = kerastuner.RandomSearch(
        _build_keras_model,
        max_trials=configs['training_and_tuning']['max_tuner_trials'],
        hyperparameters=_get_hyperparameters(),
        allow_new_entries=False,
        objective=kerastuner.Objective('val_recall', 'max'),
        directory=fn_args.working_dir,
        project_name='nbo_tuning'
    )

    transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)


    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        transform_graph,
        batch_size=configs['model_configs']['batch_size']
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        transform_graph,
        batch_size=configs['model_configs']['batch_size']
    )

    return tfx.components.TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )


def run_fn(fn_args: tfx.components.FnArgs):

    """Train the model based on given args.
    Args:
        fn_args: Holds args as name/value pairs. See
        https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - data_accessor: Contains factories that can create tf.data.Datasets or
            other means to access the train/eval data. They provide a uniform way of
            accessing data, regardless of how the data is stored on disk.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - transform_output: A uri to a path containing statistics and metadata
            from TFTransform component. produced by TFT. Will be None if not
            specified.
        - model_run_dir: A single uri for the output directory of model training
            related files.
        - hyperparameters: An optional kerastuner.HyperParameters config.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=configs['model_configs']['batch_size']
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=configs['model_configs']['batch_size']
    )

    if fn_args.hyperparameters:
        hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hparams = _get_hyperparameters()
    logging.info('Hyperparameters for training %s' % hparams.get_config())

    mirrord_strategy = tf.distribute.MirroredStrategy()

    with mirrord_strategy.scope():
        base_model = _build_keras_model(hparams=hparams)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch'
    )

    base_model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        epochs=configs['model_configs']['epochs'],
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
    )
    
    label_layer = predictionsPostprocessing(vocabulary, 5)(base_model.output)
    label_model = tf.keras.Model(base_model.input, label_layer)
    
    # Generate the serving input spec for XAI
    xai_input_spec = {}

    # -> TO_DO): Check on removing this
    for i in base_model.inputs:
        if i.dtype == tf.int32:
            xai_input_spec[i.name] = tf.TensorSpec(shape=None, dtype=tf.int64, name=i.name)
        else:
            xai_input_spec[i.name] = tf.TensorSpec(shape=None, dtype=i.dtype, name=i.name)

    # Define m_call function for XAI
    m_call = tf.function(base_model.call).get_concrete_function(xai_input_spec)

    # Get serving and transform signatures
    signatures = {
        'serving_default':
          _get_tf_examples_serving_signature(base_model, tf_transform_output),
        'transform_features':
          _get_transform_features_signature(base_model, tf_transform_output),
        'xai_preprocess':
            _get_xai_preprocess_fn(base_model, tf_transform_output),
        'xai_model':
            m_call
    }

    # Save model with signatures
    label_model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )