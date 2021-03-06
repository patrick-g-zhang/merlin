#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.contrib.layers import dropout
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import MultiRNNCell,RNNCell, BasicRNNCell, BasicLSTMCell,GRUCell, LayerNormBasicLSTMCell, DropoutWrapper,\
ResidualWrapper
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops,math_ops
import pdb
import logging


class TensorflowModels(object):

    def __init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type="linear", dropout_rate=0, loss_function="mse", optimizer="adam", initial_learning_rate=0.001):

        # special case for additional label
        self.n_in = int(n_in) - 1
        self.n_out = int(n_out)

        self.n_layers = len(hidden_layer_size)

        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_type = hidden_layer_type

        assert len(self.hidden_layer_size) == len(self.hidden_layer_type)

        self.output_type   = output_type
        self.dropout_rate  = dropout_rate
        self.loss_function = loss_function
        self.optimizer     = optimizer
        self.graph = tf.Graph()
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate

    def define_feedforward_model_utt(self):
        """
            utterance index embedding
            last dim of input should be index
            TO DO LIST:
            embedding matrix size is fixed not fit to data
        """
        layer_list = []
        with self.graph.as_default() as g:
            self.global_step = tf.Variable(0, trainable=False)
            self.is_training_batch = tf.placeholder(tf.bool, shape=(), name="is_training_batch")
            # bn_params={"is_training":is_training_batch,"decay":0.99,"updates_collections":None}
            # g.add_to_collection("is_training_batch", is_training_batch)

            with tf.name_scope("input"):
                # shape (N, 319)
                self.input_lin_layer = tf.placeholder(dtype=tf.float32, shape=(None, self.n_in), name="input_layer")
                # embedding shape (UTT, 10)
                self.utt_embeddings = tf.get_variable("utt-embeddings", [1000, 10], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
                # label (N, 1)
                self.utt_index_t = tf.placeholder(dtype=tf.int32, shape=(None, 1), name="utt_index")

                # embedding result (N, 1, 10)
                embedding_utt = tf.nn.embedding_lookup(self.utt_embeddings, self.utt_index_t)
                # concatenate embedding result and linguistic feature , shape (N, 329)
                # shape (N, 10)
                embedding_utt = tf.squeeze(embedding_utt, axis=-2)
                self.input_layer = tf.concat([self.input_lin_layer, embedding_utt], 1)

                if self.dropout_rate != 0.0:
                    print("Using dropout to avoid overfitting and the dropout rate is", self.dropout_rate)
                    is_training_drop = tf.placeholder(dtype=tf.bool, shape=(), name="is_training_drop")
                    input_layer_drop = dropout(self.input_layer, self.dropout_rate, is_training=is_training_drop)
                    layer_list.append(input_layer_drop)
                    g.add_to_collection(name="is_training_drop", value=is_training_drop)
                else:
                    layer_list.append(self.input_layer)

            # hidden layer
            for i in range(len(self.hidden_layer_size)):
                with tf.name_scope("hidden_layer_" + str(i + 1)):
                    if self.dropout_rate != 0.0:
                        last_layer = layer_list[-1]
                        if self.hidden_layer_type[i] == "tanh":
                            new_layer=fully_connected(last_layer, self.hidden_layer_size[i], activation_fn=None)
                            new_layer = tf.contrib.layers.batch_norm(new_layer,is_training=self.is_training_batch)
                            new_layer = tf.nn.tanh(new_layer)
                        if self.hidden_layer_type[i]=="sigmoid":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.sigmoid)
                        if self.hidden_layer_type[i]=="relu":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.relu)
                        if self.hidden_layer_type[i]=="selu":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.selu)
                        new_layer_drop=dropout(new_layer,self.dropout_rate,is_training=is_training_drop)
                        layer_list.append(new_layer_drop)
                    else:
                        # pdb.set_trace()
                        last_layer = layer_list[-1]
                        if self.hidden_layer_type[i] == "tanh":
                            new_layer = fully_connected(last_layer, self.hidden_layer_size[i], activation_fn=None)
                            new_layer = tf.nn.tanh(new_layer)
                            tf.summary.histogram("%s th layer activation" % str(i), new_layer)
                        if self.hidden_layer_type[i] == "sigmoid":
                            new_layer = fully_connected(last_layer, self.hidden_layer_size[i], activation_fn=tf.nn.sigmoid)
                        if self.hidden_layer_type[i] == "relu":
                            new_layer = fully_connected(last_layer, self.hidden_layer_size[i], activation_fn=tf.nn.relu)
                        if self.hidden_layer_type[i] == "selu":
                            new_layer = fully_connected(last_layer, self.hidden_layer_size[i], activation_fn=tf.nn.selu)
                        layer_list.append(new_layer)

            with tf.name_scope("output_layer"):
                if self.output_type == "linear":
                    self.output_layer = fully_connected(layer_list[-1], self.n_out, activation_fn=None)
                if self.output_type == "tanh":
                    self.output_layer = fully_connected(layer_list[-1], self.n_out, activation_fn=tf.nn.tanh)
                # g.add_to_collection(name="output_layer", value=output_layer)

    def define_feedforward_model(self):
        """
            the basic deep feedforward dnn model
        """
        layer_list=[]
        with self.graph.as_default() as g:
            #pdb.set_trace()
            self.global_step  = tf.Variable(0,trainable=False)
            g.add_to_collection(name='global_step',value=self.global_step)
            is_training_batch=tf.placeholder(tf.bool,shape=(),name="is_training_batch")
            # bn_params={"is_training":is_training_batch,"decay":0.99,"updates_collections":None}
            g.add_to_collection("is_training_batch",is_training_batch)
            with tf.name_scope("input"):
                input_layer=tf.placeholder(dtype=tf.float32,shape=(None,self.n_in),name="input_layer")
                if self.dropout_rate!=0.0:
                    print("Using dropout to avoid overfitting and the dropout rate is",self.dropout_rate)
                    is_training_drop=tf.placeholder(dtype=tf.bool,shape=(),name="is_training_drop")
                    input_layer_drop=dropout(input_layer,self.dropout_rate,is_training=is_training_drop)
                    layer_list.append(input_layer_drop)
                    g.add_to_collection(name="is_training_drop",value=is_training_drop)
                else:
                    layer_list.append(input_layer)
            g.add_to_collection("input_layer",layer_list[0])
           
            for i in range(len(self.hidden_layer_size)):
                with tf.name_scope("hidden_layer_"+str(i+1)):
                    if self.dropout_rate!=0.0:
                        last_layer=layer_list[-1]
                        if self.hidden_layer_type[i]=="tanh":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=None)
                            new_layer = tf.contrib.layers.batch_norm(new_layer,is_training=is_training_batch)
                            new_layer = tf.nn.tanh(new_layer)
                        if self.hidden_layer_type[i]=="sigmoid":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.sigmoid)
                        if self.hidden_layer_type[i]=="relu":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.relu)
                        if self.hidden_layer_type[i]=="selu":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.selu)
                        new_layer_drop=dropout(new_layer,self.dropout_rate,is_training=is_training_drop)
                        layer_list.append(new_layer_drop)
                    else:
                        # pdb.set_trace()
                        last_layer=layer_list[-1]
                        if self.hidden_layer_type[i]=="tanh":
                            new_layer = fully_connected(last_layer, self.hidden_layer_size[i], activation_fn=None)
                            # new_layer = tf.contrib.layers.batch_norm(new_layer, is_training=is_training_batch)
                            new_layer = tf.nn.tanh(new_layer)
                        if self.hidden_layer_type[i]=="sigmoid":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.sigmoid)
                        if self.hidden_layer_type[i]=="relu":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.relu)
                        if self.hidden_layer_type[i]=="selu":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.selu)
                        layer_list.append(new_layer)
            # pdb.set_trace()
            with tf.name_scope("output_layer"):
                if self.output_type=="linear":
                    output_layer=fully_connected(layer_list[-1],self.n_out,activation_fn=None)
                if self.output_type=="tanh":
                    output_layer=fully_connected(layer_list[-1],self.n_out,activation_fn=tf.nn.tanh)
                g.add_to_collection(name="output_layer",value=output_layer)


    def define_feedforward_multitask2_model(self):
        # there will be two output layers, and only update one output layer once
        # since this is for two cantonese corpus now the dimensions of input layers of two corpus are same temporarily  
        layer_list=[]
        with self.graph.as_default() as g:
            is_training_batch=tf.placeholder(tf.bool,shape=(),name="is_training_batch")
            bn_params={"is_training":is_training_batch,"decay":0.99,"updates_collections":None}
            g.add_to_collection("is_training_batch",is_training_batch)
            with tf.name_scope("input"):
                input_layer=tf.placeholder(dtype=tf.float32,shape=(None,self.n_in),name="input_layer")
                if self.dropout_rate!=0.0:
                    print("Using dropout to avoid overfitting and the dropout rate is",self.dropout_rate)
                    is_training_drop=tf.placeholder(dtype=tf.bool,shape=(),name="is_training_drop")
                    input_layer_drop=dropout(input_layer,self.dropout_rate,is_training=is_training_drop)
                    layer_list.append(input_layer_drop)
                    g.add_to_collection(name="is_training_drop",value=is_training_drop)
                else:
                    layer_list.append(input_layer)
            g.add_to_collection("input_layer",layer_list[0])
            for i in range(len(self.hidden_layer_size)):
                with tf.name_scope("hidden_layer_"+str(i+1)):
                    last_layer=layer_list[-1]
                    new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.tanh,normalizer_fn=batch_norm,\
                                normalizer_params=bn_params)
                    new_layer_drop=dropout(new_layer,self.dropout_rate,is_training=is_training_drop)
                    layer_list.append(new_layer_drop)
            with tf.name_scope("output_layer"):
                # this is the output layer for the corpus 1
                bottleneck_layer_c1 = fully_connected(layer_list[-1], 256, activation_fn=tf.nn.tanh,normalizer_fn=batch_norm,normalizer_params=bn_params)
                new_layer_drop=dropout(bottleneck_layer_c1,self.dropout_rate,is_training=is_training_drop)
                output_layer_c1=fully_connected(new_layer_drop,self.n_out,activation_fn=None)
                bottleneck_layer_c2 = fully_connected(layer_list[-1],256,activation_fn=tf.nn.tanh,normalizer_fn=batch_norm,normalizer_params=bn_params)
                new_layer_drop=dropout(bottleneck_layer_c2,self.dropout_rate,is_training=is_training_drop)
                output_layer_c2 = fully_connected(new_layer_drop,self.n_out,activation_fn=None)
                g.add_to_collection(name="output_layer_c1",value=output_layer_c1)
                g.add_to_collection(name="output_layer_c2",value=output_layer_c2)
            with tf.name_scope("training_op"):
                self.training_op=tf.train.AdamOptimizer()
    
    def define_feedforward_multitask_model(self):
        layer_list=[]
        with self.graph.as_default() as g:
            is_training_batch=tf.placeholder(tf.bool,shape=(),name="is_training_batch")
            bn_params={"is_training":is_training_batch,"decay":0.99,"updates_collections":None}
            g.add_to_collection("is_training_batch",is_training_batch)
            with tf.name_scope("input"):
                input_layer=tf.placeholder(dtype=tf.float32,shape=(None,self.n_in),name="input_layer")
                if self.dropout_rate!=0.0:
                    print("Using dropout to avoid overfitting and the dropout rate is",self.dropout_rate)
                    is_training_drop=tf.placeholder(dtype=tf.bool,shape=(),name="is_training_drop")
                    input_layer_drop=dropout(input_layer,self.dropout_rate,is_training=is_training_drop)
                    layer_list.append(input_layer_drop)
                    g.add_to_collection(name="is_training_drop",value=is_training_drop)
                else:
                    layer_list.append(input_layer)
            g.add_to_collection("input_layer",layer_list[0])
            for i in range(len(self.hidden_layer_size)):
                with tf.name_scope("hidden_layer_"+str(i+1)):
                    if self.dropout_rate!=0.0:
                        last_layer=layer_list[-1]
                        if self.hidden_layer_type[i]=="tanh":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.tanh,normalizer_fn=batch_norm,\
                                normalizer_params=bn_params)
                        if self.hidden_layer_type[i]=="sigmoid":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.sigmoid,normalizer_fn=batch_norm,\
                                normalizer_params=bn_params)
                        new_layer_drop=dropout(new_layer,self.dropout_rate,is_training=is_training_drop)
                        layer_list.append(new_layer_drop)
                    else:
                        last_layer=layer_list[-1]
                        if self.hidden_layer_type[i]=="tanh":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.tanh,normalizer_fn=batch_norm,\
                               normalizer_params=bn_params)
                        if self.hidden_layer_type[i]=="sigmoid":
                            new_layer=fully_connected(last_layer,self.hidden_layer_size[i],activation_fn=tf.nn.sigmoid,normalizer_fn=batch_norm,\
                               normalizer_params=bn_params)
                        layer_list.append(new_layer)
            with tf.name_scope("output_layer"):
                output_layer_regress=fully_connected(layer_list[-1],self.n_out,activation_fn=None)
                output_layer_classification = fully_connected(layer_list[-1],1,activation_fn=tf.nn.sigmoid)
                g.add_to_collection(name="output_layer_regress",value=output_layer_regress)
                g.add_to_collection(name="output_layer_classification",value=output_layer_classification)
            with tf.name_scope("training_op"):
                if self.optimizer=="adam":
                    self.training_op=tf.train.AdamOptimizer()

    def define_sequence_model(self):
        logger = logging.getLogger("define a sequential model")
        layer_list=[]
        with self.graph.as_default() as g:
            # the utterance lengths of this sequential data
            utt_length=tf.placeholder(tf.int32,shape=(None))
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                            global_step=global_step,
                                                            decay_steps=50000, decay_rate=0.99)

            g.add_to_collection(name="utt_length",value=utt_length)
            g.add_to_collection(name="global_step",value=global_step)
            g.add_to_collection(name="learning_rate",value=learning_rate)
            with tf.name_scope("input"):
                input_layer=tf.placeholder(dtype=tf.float32,shape=(None,None,self.n_in),name="input_layer")
                layer_list.append(input_layer)
            g.add_to_collection("input_layer",layer_list[0])
            with tf.name_scope("hidden_layer"):
                basic_cell=[]
                if "tanh" in self.hidden_layer_type:
                    is_training_batch=tf.placeholder(dtype=tf.bool,shape=(),name="is_training_batch")
                    bn_params={"is_training":is_training_batch,"decay":0.99,"updates_collections":None}
                    g.add_to_collection("is_training_batch",is_training_batch)
                for i in range(len(self.hidden_layer_type)):
                        if self.hidden_layer_type[i]=="tanh":
                            new_layer=fully_connected(layer_list[-1],self.hidden_layer_size[i],activation_fn=tf.nn.tanh,normalizer_fn=batch_norm,normalizer_params=bn_params)
                            layer_list.append(new_layer)
                        if self.hidden_layer_type[i] == "selu":
                            new_layer=fully_connected(layer_list[-1],self.hidden_layer_size[i],activation_fn=tf.nn.selu)
                            layer_list.append(new_layer)
                        if self.hidden_layer_type[i]=="lstm":
                            basic_cell.append(LayerNormBasicLSTMCell(num_units=self.hidden_layer_size[i]))
                        if self.hidden_layer_type[i]=="gru":
                            basic_cell.append(LayerNormGRUCell(num_units=self.hidden_layer_size[i]))
                multi_cell=MultiRNNCell(basic_cell)
                rnn_outputs,rnn_states=tf.nn.dynamic_rnn(multi_cell,layer_list[-1],dtype=tf.float32,sequence_length=utt_length)
                layer_list.append(rnn_outputs)
            with tf.name_scope("output_layer"):
                if self.output_type=="linear" :
                    output_layer=tf.layers.dense(rnn_outputs,self.n_out)
                g.add_to_collection(name="output_layer",value=output_layer)
            # with tf.name_scope("training_op"):
            #     self.training_op=tf.train.AdamOptimizer(learning_rate)

    def get_max_step(self,max_step):
       ##This method is only used when a sequence model is TrainTensorflowModels
        self.max_step=max_step

class MyDropoutWrapper(DropoutWrapper):

    def __init__(self, cell, is_training,input_keep_prob=1.0, output_keep_prob=1.0,
               state_keep_prob=1.0, variational_recurrent=False,
                input_size=None, dtype=None, seed=None):
        DropoutWrapper.__init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               state_keep_prob=1.0, variational_recurrent=False,
               input_size=None, dtype=None, seed=None)
        self.is_training=is_training

    def __call__(self, inputs, state, scope=None):

        return tf.cond(self.is_training,\
                    lambda: DropoutWrapper(self._cell,self._input_keep_prob,self._output_keep_prob).__call__(inputs,state,scope=None),\
                    lambda: DropoutWrapper(self._cell,1.0,1.0).__call__(inputs,state,scope=None))
           #return self._cell(dropout(inputs,self._input_keep_prob,is_training=self.is_training,scope=None),state,scope=None)

class Encoder_Decoder_Models(TensorflowModels):
        def __init__(self,n_in,encoder_layer_size,n_out,encoder_layer_type,
                     output_type="linear",dropout_rate=0,loss_function="mse",
                     optimizer="adam",attention=False,cbhg=False):
            TensorflowModels.__init__(self,n_in,encoder_layer_size,n_out,encoder_layer_type,output_type="linear",dropout_rate=0,loss_function="mse",optimizer="adam")
            self.encoder_layer_size=self.hidden_layer_size
            self.encoder_layer_type=self.hidden_layer_type
            self.attention=attention
            self.cbhg=cbhg

        def convbank(self,inputs,conv_bank_size,scope="convbank"):
            with tf.variable_scope(scope,reuse=None):
               outputs=tf.layers.conv1d(inputs,self.n_in//2,1)
               for k in range(2,conv_bank_size+1):
                    with tf.variable_scope("num_{0}".format(k)):
                        k_output=tf.layers.conv1d(inputs,self.n_in//2,k,padding="same",activation=tf.nn.relu)
                        outputs=tf.concat((outputs,k_output),-1)
            return outputs

        def pooling(self,conv_outputs,pooling_window,stride,scope="pooling"):
            with tf.variable_scope(scope,reuse=None):
                pooling_outputs=tf.layers.max_pooling1d(conv_outputs,pooling_window,stride)
        #    print pooling_outputs.shape
            return pooling_outputs

        def convproject(self,inputs,filters,width,scope="convproject"):
            with tf.variable_scope(scope,reuse=None):
                projection_layer=tf.layers.conv1d(inputs,filters,width,padding="same",activation=tf.nn.relu)
          # print projection_layer.shape
            return projection_layer

        def deep_feedfoward(self,project_outputs,num_units,layers=4,scope="feedforward"):
            with tf.variable_scope(scope,reuse=None):
                layer_list=[project_outputs]
                for l in range(layers):
                    layer_list.append(fully_connected(layer_list[-1],num_units,activation_fn=tf.nn.relu))
        #  print layer_list[-1].shape
            return layer_list[-1]

        def encoder(self,inputs,inputs_sequence_length):
            with tf.variable_scope("encoder"):
                basic_cell=[]
                for i in range(len(self.hidden_layer_size)):
                    if self.hidden_layer_type[i]=="tanh":
                        basic_cell.append(tf.contrib.rnn.BasicRNNCell(num_units=self.encoder_layer_size[i]))
                    if self.hidden_layer_type[i]=="lstm":
                        basic_cell.append(tf.contrib.rnn.BasicLSTMCell(num_units=self.encoder_layer_size[i]))
                    if self.hidden_layer_type[i]=="gru":
                         basic_cell.append(GRUCell(num_units=self.encoder_layer_size[i]))
                multicell=MultiRNNCell(basic_cell)
                enc_output, enc_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=multicell,cell_bw=multicell,inputs=inputs,\
                             sequence_length=inputs_sequence_length,dtype=tf.float32)
                enc_output=tf.concat(enc_output,2)
                #enc_state=(tf.concat(enc_state[0])
            return enc_output, enc_state

        def process_decoder_input(self,target_sequence):
            decode_input=tf.concat((tf.zeros_like(target_sequence[:, :1, :]), target_sequence[:, :-1, :]), 1)
            return decode_input

        def decoder(self,decoder_inputs,enc_output,enc_states,target_sequence_length):
            with tf.variable_scope("decoder"):
                basic_cell=[]
                for i in range(len(self.hidden_layer_size)):
                    if self.hidden_layer_type[i]=="tanh":
                        basic_cell.append(tf.contrib.rnn.BasicRNNCell(num_units=self.encoder_layer_size[i]))
                    if self.hidden_layer_type[i]=="lstm":
                        basic_cell.append(tf.contrib.rnn.BasicLSTMCell(num_units=self.encoder_layer_size[i]))
                    if self.hidden_layer_type[i]=="gru":
                         basic_cell.append(GRUCell(num_units=self.encoder_layer_size[i]))
                multicell=MultiRNNCell(basic_cell)
            if not self.attention:
                dec_output,_=tf.nn.bidirectional_dynamic_rnn(cell_fw=multicell,cell_bw=multicell,inputs=decoder_inputs,
                                                             initial_state_fw=enc_states[0], sequence_length=target_sequence_length,
                                                             initial_state_bw=enc_states[1])
            else:
                  attention_size=decoder_inputs.get_shape().as_list()[-1]
                  attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(attention_size,enc_output,target_sequence_length,normalize=True,probability_fn=tf.nn.softmax)
                  cell_with_attention=tf.contrib.seq2seq.AttentionWrapper(multicell,attention_mechanism,attention_size)
                  dec_output,_=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_with_attention,cell_bw=cell_with_attention,inputs=decoder_inputs,dtype=tf.float32)
            return dec_output

        def define_encoder_decoder(self):
            with self.graph.as_default() as g:
                with tf.name_scope("encoder_input"):
                    inputs_data=tf.placeholder(dtype=tf.float32,shape=[None,None,self.n_in],name="inputs_data")
                    if self.cbhg:
                        conv_bank=self.convbank(inputs_data,16)
                        max_pooled_=self.pooling(conv_bank,2,1)
                        conv_project=self.convproject(max_pooled_,self.n_in//2,3)
                        encoder_inputs=self.deep_feedfoward(conv_project,self.n_in//2,4)
                    else:
                        inputs_sequence_length=tf.placeholder(tf.int32,shape=[None],name="inputs_sequence_length")
                        g.add_to_collection("inputs_sequence_length",inputs_sequence_length)
                    g.add_to_collection("inputs_data",inputs_data)
                with tf.name_scope("target_sequence"):
                   targets=tf.placeholder(dtype=tf.float32,shape=[None,None,self.n_out],name="targets")
                   target_sequence_length=tf.placeholder(tf.int32,[None],name="target_sequence_length")
                   g.add_to_collection("targets",targets)
                   g.add_to_collection("target_sequence_length",target_sequence_length)

                with tf.name_scope("encoder_output"):
                    if self.cbhg:
                        enc_out,enc_states=self.encoder(encoder_inputs,None)
                    else:
                        enc_out,enc_states=self.encoder(inputs_data,inputs_sequence_length)
                with tf.name_scope("decoder_inputs"):
                    dec_inputs=self.process_decoder_input(targets)
                with tf.name_scope("decoder_outputs"):
                    dec_output=self.decoder(dec_inputs,enc_out,enc_states,target_sequence_length)
                    dec_output=tf.concat(dec_output,2)
                with tf.name_scope("outputs"):
                    if self.output_type=="linear":
                        outputs=tf.layers.dense(dec_output,self.n_out)
                    g.add_to_collection(name="decoder_outputs",value=outputs)
                with tf.name_scope("training_op"):
                    if self.optimizer=="adam":
                        self.training_op=tf.train.AdamOptimizer(0.002)

def layer_normalization(inputs,epsilon = 1e-5,scope=None):
    mean,var=tf.nn.moments(inputs,[1],keep_dims=True)
    with tf.variable_scope(scope+"LN",reuse=None):
            scale=tf.get_variable(name="scale",shape=[inputs.get_shape()[1]],initializer=tf.constant_initializer(1))
            shift=tf.get_variable(name="shift",shape=[inputs.get_shape()[1]],initializer=tf.constant_initializer(0))
    LN_output=scale*(inputs-mean)/tf.sqrt(var + epsilon) + shift
    return LN_output

#Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
class LayerNormGRUCell(RNNCell):
    def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
        super(LayerNormGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state):
        # Gated recurrent unit (GRU) with nunits cells.
        with vs.variable_scope("gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
        if self._bias_initializer is None:
            dtype = [a.dtype for a in [inputs, state]][0]
            bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
        value = rnn_cell_impl._linear([inputs, state], 2 * self._num_units, True, bias_ones,\
                  self._kernel_initializer)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        r,u=layer_normalization(r,scope="r/"),layer_normalization(u,scope="u/")
        r,u=math_ops.sigmoid(r),math_ops.sigmoid(u)
        with vs.variable_scope("candidate"):
            c = self._activation(rnn_cell_impl._linear([inputs, r * state], self._num_units, True, self._bias_initializer, self._kernel_initializer))
            new_h = u * state + (1 - u) * c
        return new_h, new_h
