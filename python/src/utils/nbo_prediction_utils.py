from copy import deepcopy
from typing import List, Optional, Text
import os
import absl

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.python.keras.engine.training import Model
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow.python.lib.io import file_io

from google.protobuf import text_format
from google.cloud import storage
from google.api_core import page_iterator

import numpy as np
import pandas as pd
from operator import itemgetter 

from src.utils.helper_utils import get_latest_version, get_tft_vocab

def get_schema_url(gcs_bucket: Text, pipeline_name:Text):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    """
    schema_folder = 'SchemaGen/schema'
    prefix = os.path.join('tfx', pipeline_name, schema_folder)
    return os.path.join(
        gcs_bucket,
        prefix,
        get_latest_version(
            bucket=gcs_bucket,
            prefix=prefix
        ),
        'schema.pbtxt'
    )


def get_model_url(gcs_bucket:Text, pipeline_name:Text):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    """

    model_folder = 'Pusher/pushed_model'
    prefix = os.path.join('tfx', pipeline_name, model_folder)
    return os.path.join(
        gcs_bucket,
        prefix,
        get_latest_version(
            bucket=gcs_bucket,
            prefix=prefix
        )
    )


def get_model(model_url:Text):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    """
    model = tf.keras.models.load_model(model_url)
    absl.logging(model.summary())
    return model


def get_inference_fun(model:tf.keras.Model, 
                    signature_name:Optional[Text]='serving_default'):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    """

    return model.signatures[signature_name]


def read_schema(path:Text):
    """
    Description
    -----------
    Reads a schema from the provided location.

    Parameters
    ----------
    path (Text): The location of the file holding a serialized Schema proto.

    Returns
    -------
    An instance of Schema or None if the input argument is None
    """

    result = schema_pb2.Schema()
    contents = file_io.read_file_to_string(path)
    text_format.Parse(contents, result)

    return result


def get_raw_feature_spec(schema):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    
    """

    return schema_utils.schema_as_feature_spec(schema).feature_spec


def make_proto_coder(schema):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------

    """

    raw_feature_spec = get_raw_feature_spec(schema)
    raw_schema = schema_utils.schema_as_feature_spec(raw_feature_spec)
    
    return tft_coders.ExampleProtoCoder(raw_schema)


def remove_target(raw_feature_spec, label_key):
    return raw_feature_spec.pop(label_key)


def prepare_data_for_requests(input_data, schema, label_key):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    
    """

    schema_copy = deepcopy(schema)

    # Make a copy of the schema, so we can safely pop the label
    filtered_features = [
          feature for feature in schema_copy.feature if feature.name != label_key
      ]

    del schema_copy.feature[:]
    schema_copy.feature.extend(filtered_features)

    proto_coder = make_proto_coder(schema_copy)

    # Make a copy of the input data, so we can safely pop the label
    input_copy = input_data.copy()

    # remove the label
    input_copy.pop(label_key)

    # Create an empty dict to store serialized examples.
    features = {}

    for feature in schema_copy.feature:
        name = feature.name
        if feature.type == schema_pb2.FLOAT:
            features[name] = tf.train.Feature(float_list=tf.train.FloatList(value=list(input_data[name])))
        elif feature.type == schema_pb2.INT:
            features[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_data[name])))
        elif feature.type == schema_pb2.BYTES:
            features[name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(input_data[name].str.encode('utf-8'))))
        else:
            features[name] = []

    absl.logging(features)

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))

    examples = example_proto.SerializeToString()

    return examples


def get_prediction(inference_fn, examples):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    
    """
    return inference_fn(examples=tf.constant([examples]))



def get_product_recommendations(predictions, label_name, bucket, pipeline_name, pipeline_root, top_k=3):
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    
    """

    vocab = get_tft_vocab(
        name=label_name,
        bucket=bucket,
        directory=os.path.join('tfx', pipeline_name, 'Transform/transform_graph'),
        pipeline_root=pipeline_root,
        full_vocab=True
    )

    top_k_ind = np.argpartition(predictions, -top_k)[-top_k:]

    return itemgetter(*top_k_ind)(vocab)