import os
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python import debug as tfdbg
from utils import createVocabulary
from utils import loadVocabulary
from utils import computeF1Score
from utils import DataProcessor

parser = argparse.ArgumentParser(allow_abbrev=False)
##
#Network
parser.add_argument("--num_units", type=int, default=64, help="Network size.", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""full(default) | intent_only
                                                                    full: full attention model
                                                                    intent_only: intent attention model""")

#Training Environment
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false',dest='early_stop', help="Disable early stop, which is based on sentence level accuracy.")
parser.add_argument("--patience", type=int, default=5, help="Patience to wait before stop.")

#Model and Vocab
parser.add_argument("--dataset", type=str, default=None, help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")

#Data
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

arg=parser.parse_args()

#Print arguments
for k,v in sorted(vars(arg).items()):
    print(k,'=',v)
print()

if arg.model_type == 'full':
    add_final_state_to_intent = True
    remove_slot_attn = False
elif arg.model_type == 'intent_only':
    add_final_state_to_intent = True
    remove_slot_attn = True
else:
    print('unknown model type!')
    exit(1)

#full path to data will be: ./data + dataset + train/test/valid
if arg.dataset == None:
    print('name of dataset can not be None')
    exit(1)
elif arg.dataset == 'snips':
    print('use snips dataset')
elif arg.dataset == 'atis':
    print('use atis dataset')
else:
    print('use own dataset: ',arg.dataset)
full_train_path = os.path.join('./data',arg.dataset,arg.train_data_path)
full_test_path = os.path.join('./data',arg.dataset,arg.test_data_path)
full_valid_path = os.path.join('./data',arg.dataset,arg.valid_data_path)

createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, 'in_vocab'))
createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, 'slot_vocab'))
createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, 'intent_vocab'))

in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))

def createModel(input_data, input_size, sequence_length, slot_size, intent_size, layer_size = 128, isTraining = True):
    cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)

    if isTraining == True:
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5,
                                             output_keep_prob=0.5)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5,
                                             output_keep_prob=0.5)

    embedding = tf.get_variable('embedding', [input_size, layer_size])
    inputs = tf.nn.embedding_lookup(embedding, input_data)
    # State_outputs: ontains forward and backwards sequence. Shape: 2 x batchsize x len x dim
    # Final_state: The final states of both forward and backwards LSTM. Shape: 2 x 2(cell and hidden) x batchsize x dim
    state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=sequence_length, dtype=tf.float32)
    
    # concatenate in the last dim, so final will become batch_size x dim(256)
    final_state = tf.concat([final_state[0][0], final_state[0][1], final_state[1][0], final_state[1][1]], 1)
    state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2) # Shape: batchsize x len x dim(128)
    state_shape = state_outputs.get_shape()
    
    with tf.variable_scope('attention'):
        slot_inputs = state_outputs
        if remove_slot_attn == False:
            with tf.variable_scope('slot_attn'):
                attn_size = state_shape[2].value # dim(128)
                origin_shape = tf.shape(state_outputs)
                hidden = tf.expand_dims(state_outputs, 1) # Shape: batchsize x 1 x len x dim(128)
                hidden_conv = tf.expand_dims(state_outputs, 2) # Shape: batchsize x len x 1 x dim(128)
                # hidden shape = [batch, sentence length, 1, hidden size]
                k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size]) # 1 x 1 x 128 x 128
                # Convolutional: Attention weights
                hidden_features = tf.nn.conv2d(hidden_conv, k, [1, 1, 1, 1], "SAME")
                hidden_features = tf.reshape(hidden_features, origin_shape)
                hidden_features = tf.expand_dims(hidden_features, 1)
                # Derive the hidden states weighted from attention (Content vector)
                v = tf.get_variable("AttnV", [attn_size])# 128

                slot_inputs_shape = tf.shape(slot_inputs)
                slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])# Shape: (batchsize x len) x dim(128)
                y = _linear(slot_inputs, attn_size, True)# The y here is the origin hidden states.
                y = tf.reshape(y, slot_inputs_shape)
                y = tf.expand_dims(y, 2)
                s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])# Sum the origin hidden states and weighted hidden states up
                a = tf.nn.softmax(s)
                # a shape = [batch, input size, sentence length, 1]
                a = tf.expand_dims(a, -1)
                slot_d = tf.reduce_sum(a * hidden, [2])
        else:
            attn_size = state_shape[2].value
            slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])

        intent_input = final_state # Shape: 2 x 2(cell and hidden) x batchsize x dim
        with tf.variable_scope('intent_attn'):
            attn_size = state_shape[2].value # dim(128)
            hidden = tf.expand_dims(state_outputs, 2) # Shape: batchsize x len x 1 x dim(128)
            k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size]) # 1 x 1 128 x 128
            # Attention weighted
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV", [attn_size])

            y = _linear(intent_input, attn_size, True)
            y = tf.reshape(y, [-1, 1, 1, attn_size])
            s = tf.reduce_sum(v*tf.tanh(hidden_features + y), [2,3])
            a = tf.nn.softmax(s)
            a = tf.expand_dims(a, -1)
            a = tf.expand_dims(a, -1)
            d = tf.reduce_sum(a * hidden, [1, 2])

            if add_final_state_to_intent == True:
                intent_output = tf.concat([d, intent_input], 1)
            else:
                intent_output = d

        with tf.variable_scope('slot_gated'):
            intent_gate = _linear(intent_output, attn_size, True) 
            intent_gate = tf.reshape(intent_gate, [-1, 1, intent_gate.get_shape()[1].value])
            v1 = tf.get_variable("gateV", [attn_size])
            if remove_slot_attn == False:
                slot_gate = v1 * tf.tanh(slot_d + intent_gate)
            else:
                slot_gate = v1 * tf.tanh(state_outputs + intent_gate)
            slot_gate = tf.reduce_sum(slot_gate, [2])
            slot_gate = tf.expand_dims(slot_gate, -1)
            if remove_slot_attn == False:
                slot_gate = slot_d * slot_gate
            else:
                slot_gate = state_outputs * slot_gate
            slot_gate = tf.reshape(slot_gate, [-1, attn_size])
            slot_output = tf.concat([slot_gate, slot_inputs], 1)

    with tf.variable_scope('intent_proj'):
        intent = _linear(intent_output, intent_size, True)

    with tf.variable_scope('slot_proj'):
        slot = _linear(slot_output, slot_size, True)

    outputs = [slot, intent, sa]
    return outputs

# Create Training Model
input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
global_step = tf.Variable(0, trainable=False, name='global_step')
slots = tf.placeholder(tf.int32, [None, None], name='slots')
slot_weights = tf.placeholder(tf.float32, [None, None], name='slot_weights')
intent = tf.placeholder(tf.int32, [None], name='intent')

with tf.variable_scope('model'):
    training_outputs = createModel(input_data, len(in_vocab['vocab']), sequence_length, len(slot_vocab['vocab']), len(intent_vocab['vocab']), layer_size=arg.layer_size)

slots_shape = tf.shape(slots)
slots_reshape = tf.reshape(slots, [-1])
# Debug print
sa = training_outputs[2]
slot_outputs = training_outputs[0]
with tf.variable_scope('slot_loss'):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slots_reshape, logits=slot_outputs)
    crossent = tf.reshape(crossent, slots_shape)
    slot_loss = tf.reduce_sum(crossent*slot_weights, 1)
    total_size = tf.reduce_sum(slot_weights, 1)
    total_size += 1e-12
    slot_loss = slot_loss / total_size

intent_output = training_outputs[1]
with tf.variable_scope('intent_loss'):
    crossent =tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent, logits=intent_output)
    intent_loss = tf.reduce_sum(crossent) / tf.cast(arg.batch_size, tf.float32)

params = tf.trainable_variables()
opt = tf.train.AdamOptimizer()

intent_params = []
slot_params = []
for p in params:
    if not 'slot_' in p.name:
        intent_params.append(p)
    if 'slot_' in p.name or 'bidirectional_rnn' in p.name or 'embedding' in p.name:
        slot_params.append(p)

gradients_slot = tf.gradients(slot_loss, slot_params)
gradients_intent = tf.gradients(intent_loss, intent_params)

clipped_gradients_slot, norm_slot = tf.clip_by_global_norm(gradients_slot, 5.0)
clipped_gradients_intent, norm_intent = tf.clip_by_global_norm(gradients_intent, 5.0)

gradient_norm_slot = norm_slot
gradient_norm_intent = norm_intent
update_slot = opt.apply_gradients(zip(clipped_gradients_slot, slot_params))
update_intent = opt.apply_gradients(zip(clipped_gradients_intent, intent_params), global_step=global_step)
# Debug output
training_outputs = [global_step, slot_loss, update_intent, update_slot, gradient_norm_intent, gradient_norm_slot, sa]
inputs = [input_data, sequence_length, slots, slot_weights, intent]

# Create Inference Model
with tf.variable_scope('model', reuse=True):
    inference_outputs = createModel(input_data, len(in_vocab['vocab']), sequence_length, len(slot_vocab['vocab']), len(intent_vocab['vocab']), layer_size=arg.layer_size, isTraining=False)

inference_slot_output = tf.nn.softmax(inference_outputs[0], name='slot_output')
inference_intent_output = tf.nn.softmax(inference_outputs[1], name='intent_output')

inference_outputs = [inference_intent_output, inference_slot_output]
inference_inputs = [input_data, sequence_length]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

saver = tf.train.Saver()

# Start Training
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    logging.info('Training Start')

    epochs = 0
    loss = 0.0
    data_processor = None
    line = 0
    num_loss = 0
    step = 0
    no_improve = 0

    #variables to store highest values among epochs, only use 'valid_err' for now
    valid_slot = 0
    test_slot = 0
    valid_intent = 0
    test_intent = 0
    valid_err = 0
    test_err = 0

    while True:
        if data_processor == None:
            data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file), os.path.join(full_train_path, arg.slot_file), os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab, intent_vocab)
        in_data, slot_data, slot_weight, length, intents,_,_,_ = data_processor.get_batch(arg.batch_size)
        feed_dict = {input_data.name: in_data, slots.name: slot_data, slot_weights.name: slot_weight, sequence_length.name: length, intent.name: intents}
        ret = sess.run(training_outputs, feed_dict)
        loss += np.mean(ret[1])

        line += arg.batch_size
        step = ret[0]
        num_loss += 1

        if data_processor.end == 1:
            line = 0
            data_processor.close()
            data_processor = None
            epochs += 1
            logging.info('Step: ' + str(step))
            logging.info('Epochs: ' + str(epochs))
            logging.info('Shape: '+ str(ret[6]))
            logging.info('Loss: ' + str(loss/num_loss))
            num_loss = 0
            loss = 0.0

            save_path = os.path.join(arg.model_path,'_step_' + str(step) + '_epochs_' + str(epochs) + '.ckpt')
            saver.save(sess, save_path)

            def valid(in_path, slot_path, intent_path):
                data_processor_valid = DataProcessor(in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab)

                pred_intents = []
                correct_intents = []
                slot_outputs = []
                correct_slots = []
                input_words = []

                #used to gate
                gate_seq = []
                while True:
                    in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)
                    feed_dict = {input_data.name: in_data, sequence_length.name: length}
                    ret = sess.run(inference_outputs, feed_dict)
                    for i in ret[0]:
                        pred_intents.append(np.argmax(i))
                    for i in intents:
                        correct_intents.append(i)

                    pred_slots = ret[1].reshape((slot_data.shape[0], slot_data.shape[1], -1))
                    for p, t, i, l in zip(pred_slots, slot_data, in_data, length):
                        p = np.argmax(p, 1)
                        tmp_pred = []
                        tmp_correct = []
                        tmp_input = []
                        for j in range(l):
                            tmp_pred.append(slot_vocab['rev'][p[j]])
                            tmp_correct.append(slot_vocab['rev'][t[j]])
                            tmp_input.append(in_vocab['rev'][i[j]])

                        slot_outputs.append(tmp_pred)
                        correct_slots.append(tmp_correct)
                        input_words.append(tmp_input)

                    if data_processor_valid.end == 1:
                        break

                pred_intents = np.array(pred_intents)
                correct_intents = np.array(correct_intents)
                accuracy = (pred_intents==correct_intents)
                semantic_error = accuracy
                accuracy = accuracy.astype(float)
                accuracy = np.mean(accuracy)*100.0

                index = 0
                for t, p in zip(correct_slots, slot_outputs):
                    # Process Semantic Error
                    if len(t) != len(p):
                        raise ValueError('Error!!')

                    for j in range(len(t)):
                        if p[j] != t[j]:
                            semantic_error[index] = False
                            break
                    index += 1
                semantic_error = semantic_error.astype(float)
                semantic_error = np.mean(semantic_error)*100.0

                f1, precision, recall = computeF1Score(correct_slots, slot_outputs)
                logging.info('slot f1: ' + str(f1))
                logging.info('intent accuracy: ' + str(accuracy))
                logging.info('semantic error(intent, slots are all correct): ' + str(semantic_error))

                data_processor_valid.close()
                return f1,accuracy,semantic_error,pred_intents,correct_intents,slot_outputs,correct_slots,input_words,gate_seq

            logging.info('Valid:')
            epoch_valid_slot, epoch_valid_intent, epoch_valid_err,valid_pred_intent,valid_correct_intent,valid_pred_slot,valid_correct_slot,valid_words,valid_gate = valid(os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.slot_file), os.path.join(full_valid_path, arg.intent_file))

            logging.info('Test:')
            epoch_test_slot, epoch_test_intent, epoch_test_err,test_pred_intent,test_correct_intent,test_pred_slot,test_correct_slot,test_words,test_gate = valid(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.slot_file), os.path.join(full_test_path, arg.intent_file))

            if epoch_valid_err <= valid_err:
                no_improve += 1
            else:
                valid_err = epoch_valid_err
                no_improve = 0

            if epochs == arg.max_epochs:
                break

            if arg.early_stop == True:
                if no_improve > arg.patience:
                    break

