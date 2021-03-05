import tensorflow as tf
import numpy as np
import pandas as pd
import json

# def get_loss_fn(num_classes, class_weights):
    # class_weights should be a *list* of length num_classes  
    # """Gets the classification loss function."""
def classification_loss_fn(labels, logits):
    """Classification loss."""
    # labels is a 1D tensor filled with numerically encoded labels (each entry corresponds to a sentence)
    labels = tf.squeeze(labels)
    # one_hot_labels is a 2D tensor (each row corresponds to a sentence)
    one_hot_labels = tf.one_hot(tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    
    # log_probs is a 2D tensor (each row corresponds to a sentence)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    # Size of the minibatch
    minibatch_size = logits.shape[0]

    # remaining_weights is a 1D tensor with length minibatch_size and is obtained by filtering the relevant class weight for each entry in the minibatch (elementwise multiplication between one_hot_labels and Wclass, both of them being 2D tensors) 
    remaining_weights = tf.reduce_sum(tf.cast(one_hot_labels, dtype=tf.float32) * class_weights, axis=-1)
   
    # Below: one_hot_labels * log_probs is a 2D tensor; the reduce_sum operation along the second dimension yields a 1D tensor (length = size of minibatch)

    # Elementwise multiplication between two 1D tensors; each element corresponds to the cross-entropy loss of a particular example 
    per_example_loss = - remaining_weights * (tf.reduce_sum(tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1))
    # weights_sum is a scalar that corresponds to the sum of the class weights for each entry in the minibatch
    weights_sum = tf.reduce_sum(remaining_weights, axis=-1)
    # Average loss in the minibatch (average_loss is a scalar)
    average_loss = tf.reduce_sum(per_example_loss, axis=-1)/weights_sum
    return average_loss
    # return classification_loss_fn

if __name__ == "__main__":
    # data_dir = f'gs://{args.bucket_name}/{args.project_name}/finetune/finetune_data/{args.finetune_data}' 
    # imbalanced_classes_location = os.path.join(data_dir, 'tr_set_imbalanced_classes_frequency.json')
    imbalanced_classes_location = 'imbalanced_classes_frequency.json'
    with tf.io.gfile.GFile(imbalanced_classes_location, 'rb') as reader:
        class_weights_dict = json.loads(reader.read().decode('utf-8'))
    
    # Argument of classification_loss_fn
    num_labels = len(class_weights_dict)
    
    # Argument of classification_loss_fn
    class_weights = list(class_weights_dict.values())

    # Labels must be numerically encoded
    string_labels = list(class_weights_dict.keys())
    unique_labels = np.unique(np.array(string_labels))
    mapping = {unique_labels[i]: i for i in range(len(unique_labels))}
    
    int_labels = list(pd.Series(string_labels).map(mapping))
    # Argument of classification_loss_fn
    labels = tf.constant(list(int_labels))
     
    # The following represents a tweak of the function run from the script run_finetune_CTBERT_tensorflow.py
# def run(args):
    # ...
    # # Get model
    # classifier_model, core_model = get_model(args, model_config, steps_per_epoch, warmup_steps, num_labels, max_seq_length)
    # optimizer = classifier_model.optimizer
    # loss_fn = get_loss_fn(num_labels, class_weights)
    
    # Argument of classification_loss_fn
    logits = tf.constant([[1,2,1],[4,5,4],[1,2,3]], dtype=tf.float32)
    loss_fn = classification_loss_fn(num_labels, class_weights, labels, logits)
    print(f'Loss: {loss_fn}')
    # ...
    
    # Next line is IMPORTANT
    # classifier_model.compile(optimizer=optimizer, loss=loss_fn, metrics=get_metrics())
    
    # Create all custom callbacks
    # ...
    # custom_callbacks = [summary_callback, time_history_callback]
    # ...
    # Generate dataset_fn
    # train_input_fn = ...
    # eval_input_fn = ...
    
    # Add metrics callback to calculate performance metrics at the end of epoch
    # performance_metrics_callback = ...
    
    # Run keras fit
    # Next line is IMPORTANT
    # history = classifier_model.fit(
    #                               x=train_input_fn(), 
    #                               validation_data=eval_input_fn(), 
    #                               steps_per_epoch=steps_per_epoch, 
    #                               epochs=args.num_epochs, 
    #                               validation_steps=eval_steps,
    #                               validation_freq=validation_freq,
    #                               callbacks=custom_callbacks,
    #                               verbose=1)
    # ...
    # ...
    # data = {...} 
    # Write run_log
    # f_path_training_log = os.path.join(output_dir, 'run_logs.json')
    # save_to_json(data, f_path_training_log)
    # Write bert config
    # ...
    # save_to_json(model_config.to_dict(), f_path_bert_config)
    # ...
# def main(args):
    # ...
    # for repeat in range(args.repeats):
    #   with strategy.scope():
    #       run(args)
