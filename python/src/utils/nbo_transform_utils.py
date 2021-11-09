import os
from typing import List, Text, Any, Optional

import numpy as np
import pandas as pd

from datetime import datetime

import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorflow_transform as tft

from src.utils.helper_utils import get_config

# Get configurations from yaml file
configs = get_config()

####################################
######## Vocabulary Configs ########
#################################### 

def get_vocabulary(vocab_root: Text, 
                column: Optional[Any]=None,
                combined_cols: Optional[bool]=False,
                **kwargs):
    """
    Description
    -----------
    Generates vocabulary list based on input file provided.

    Parameters
    ----------
    vocab_root (string): String indicating location in GCS where 
                            vocabulary csv file is stored.
    column (list): List containing column name(s) of dataframe 
                            needed to generate vocabulary
    cobmined_cols (bool): boolean indicating if vocabulary is to be created by
                            combining multiple columns in the input csv

    Returns
    -------
    Returns both the full vocabulary as well as the length of the vocabulary
    
    """

    if not combined_cols and type(column) == list:
        raise ValueError("""If combined_cols is set to False, 
        columns parameter should be single column name of type string""")

    with file_io.FileIO(vocab_root, 'r') as f:
        vocab = pd.read_csv(f)

    if combined_cols:
        vocab['joined'] = vocab[column].agg('-'.join, axis=1)
        vocabulary = vocab['joined'].tolist()
    else:
        vocabulary = vocab[column].tolist()

    return vocabulary, len(vocabulary)

_, _vocab_size = get_vocabulary(
    vocab_root=configs['data']['vocab_root_csv'],
    column='vocabulary'
)

####################################
######### Helper Functions #########
####################################

def _transformed_name(x):
    return x + '_xf'

def _fill_in_missing(x):
    """
    Replace missing values in a SparseTensor

    Fills in missing values of 'x' with '' or 0, and converts to dense tensor.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    if x.dtype == tf.string:
        default_value = ""
    elif x.dtype == tf.int32 or x.dtype == tf.int64:
        default_value = 0
    elif x.dtype == tf.float32 or x.dtype == tf.float64:
        default_value = 0.0
    else:
        raise ValueError("Could not find default missing value")
    
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value
        ),
        axis=1
    )

def _one_hot_encode(label: Any, name:Text):
    """
    Description
    -----------
    Creates one-hot encoded sequence matrices.
    Vocabulary is calculated based on input labels.
    Product names are mapped to index value based on vocabulary.

    Example
    -------
    Input_1 = ['a,b,c,d']
    -> becomes [1,1,1,1]

    Input_2 = ['a,a,b,c']
    -> becomes [1,1,1,0]

    Input_3 = ['a']
    -> becomes [1,0,0,0]
    """

    sparse_label = tf.strings.split(
        tf.reshape(
            tensor=label, 
            shape=[-1]
        ),sep=','
    ).to_sparse()

    # Apply vocabulary from txt file
    encoded_label = tft.apply_vocabulary(
        x=sparse_label,
        # -> To-DO): Make sure prediction pipeline uses same vocabulary as training pipeline
        deferred_vocab_filename_tensor=tf.constant(configs['data']['vocab_root_txt']),
        file_format='text'
    )

    label_matrix = tf.cast(
        x=tf.minimum(
            x=tf.reduce_sum(
                input_tensor=tf.nn.embedding_lookup(
                    # Generate identity matrix with size vocabulary + 2 to
                    # to correct for zero index and default value
                    np.identity(_vocab_size+1),
                    tf.sparse.to_dense(
                        sp_input=encoded_label,
                        default_value=-1
                    # add 1 to transform default values to 0
                    ) + 1 
                ),
                axis=1
            ), 
            y=1
        # remove the first value of every tensor as it is indicating a default value
        )[:, 1:],   
        dtype=tf.float32
    )

    return label_matrix


####################################
#### Preprocessing Functions #######
####################################

def preprocessing_fn(inputs):

    outputs = {}

    # Calculate vocabulary from categorical features
    for key in configs['data']['vocab_feature_keys']:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            x = _fill_in_missing(inputs[key]),
            frequency_threshold = 1, # -> TO-DO): check if this a valid value
            vocab_filename=_transformed_name(key),
            default_value=-1
        )

    # Keep boolean as is
    for key in configs['data']['bool_feature_keys']:
        outputs[_transformed_name(key)] = inputs[key]

    # Scale numerical features of type int
    for key in configs['data']['dense_int_feature_keys']:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(
                _fill_in_missing(inputs[key]),
        )

    # Scale numerical features of type float
    for key in configs['data']['dense_float_feature_keys']:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key])
        )

    # Transform string features
    for key in configs['data']['string_feature_keys']:
        outputs[_transformed_name(key)] = tf.strings.regex_replace(
            input=inputs[key], 
            pattern=',',
            rewrite=' '
        )

    # One-hot encode labels
    outputs[_transformed_name(configs['data']['target_name'])] = _one_hot_encode(
            inputs[configs['data']['target_name']], configs['data']['target_name'])

    return outputs