from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

"Adapted from the tensorflow/models/official/nlp/bert/input_pipeline.py Github"

def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example

def single_file_dataset(input_file, name_to_features):
  """Creates a single-file dataset to be passed for BERT custom training."""
  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  d = tf.data.TFRecordDataset(input_file)
  d = d.map(lambda record: decode_record(record, name_to_features))

  # When `input_file` is a path to a single file or a list
  # containing a single path, disable auto sharding so that
  # same input file is sent to all workers.
  if isinstance(input_file, str) or len(input_file) == 1:
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF)
    d = d.with_options(options)
  return d

def get_encoder_input_dict(file_path, seq_length, input_pipeline_context=None):
    """Creates input dictionary from (tf)records files for train/eval."""
    name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], tf.int64),
      'is_real_example': tf.io.FixedLenFeature([], tf.int64)
      }
    dataset = single_file_dataset(file_path, name_to_features)
    
    # The dataset is always sharded by number of hosts.
    # num_input_pipelines is the number of hosts rather than number of cores.
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_pipeline_context.num_input_pipelines, input_pipeline_context.input_pipeline_id)
      
    def _select_data_from_record(record):
        x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
        }
        return x    
    
    encoder_input_dataset = dataset.map(_select_data_from_record)
    encoder_input_df = tfds.as_dataframe(encoder_input_dataset)
    input_word_ids_list = [element.tolist() for element in encoder_input_df.input_word_ids.values]
    input_mask_list = [element.tolist() for element in encoder_input_df.input_mask.values]
    input_type_ids_list = [element.tolist() for element in encoder_input_df.input_type_ids.values]

    encoder_input_dict = {'input_word_ids': tf.constant(input_word_ids_list), 'input_mask': tf.constant(input_mask_list), 'input_type_ids': input_type_ids_list}
    return encoder_input_dict
