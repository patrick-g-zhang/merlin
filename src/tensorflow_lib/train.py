#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import random, os ,sys
from io_funcs.binary_io import BinaryIOCollection
from tensorflow_lib.model import TensorflowModels, Encoder_Decoder_Models
from tensorflow_lib import data_utils
import pdb
import logging
import random

class TrainTensorflowModels(TensorflowModels):
    def __init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, model_dir,output_type='linear', dropout_rate=0.0, loss_function='mse', optimizer='adam', rnn_params=None):

        TensorflowModels.__init__(self, n_in, hidden_layer_size, n_out, hidden_layer_type, output_type, dropout_rate, loss_function, optimizer)

        #### TODO: Find a good way to pass below params ####
        self.ckpt_dir = model_dir

    def dimension_variable_summary(self, var, dimension_list, varible_name):
        """
            show distribution of dimensions of varible
            var: the tensor of varible,assume the last dimension of shape in the dimension of tensor
            dimension_list: the dimensions you want to show
        """
        for dim in dimension_list:
            dim_summary_name = str(dim)+"th dimension of "+varible_name
            with tf.name_scope(dim_summary_name):
                tf.summary.histogram(dim_summary_name, var[:,dim])

    def train_feedforward_model(self, train_x, train_y, valid_x, valid_y, batch_size=256, num_of_epochs=10, shuffle_data=True):
        # pdb.set_trace()
        logger = logging.getLogger("train feedforward model")
        seed=12345
        np.random.seed(seed)
        print(train_x.shape)
        with self.graph.as_default() as g:
            output_data=tf.placeholder(dtype=tf.float32,shape=(None,self.n_out),name="output_data")
            input_layer=g.get_collection(name="input_layer")[0]
            is_training_batch=g.get_collection(name="is_training_batch")[0]
            if self.dropout_rate!=0.0:
                is_training_drop=g.get_collection(name="is_training_drop")[0]
            with tf.name_scope("loss"):
               output_layer=g.get_collection(name="output_layer")[0]
               loss = 0.5*tf.reduce_sum(tf.reduce_mean(tf.square(output_layer-output_data), axis=0),name="loss")
               #loss=tf.reduce_mean(tf.square(output_layer-output_data),name="loss")
            tf.summary.scalar('mean_loss', loss)
            tf.summary.scalar('learning_rate', self.learning_rate)
            merged = tf.summary.merge_all()
            with tf.name_scope("train"):
                self.training_op=self.training_op.minimize(loss,global_step=self.global_step)
            init=tf.global_variables_initializer()
            self.saver=tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter(self.ckpt_dir + '/train',
                                      sess.graph)
                test_writer = tf.summary.FileWriter(self.ckpt_dir + '/test')
                init.run()
                batch_num = int(train_x.shape[0]/batch_size)+1
                # summary_writer=tf.summary.FileWriter(os.path.join(self.ckpt_dir,"losslog"),sess.graph)
                for epoch in range(num_of_epochs):
                    L_training=0
                    L_test=0
                    overall_training_loss=0
                    overall_test_loss=0
                    for iteration in range(int(train_x.shape[0]/batch_size)+1):
                        if (iteration+1)*batch_size>train_x.shape[0]:
                            x_batch,y_batch=train_x[iteration*batch_size:],train_y[iteration*batch_size:]
                            if list(x_batch)!=[]:
                                L_training+=1
                            else:
                                continue
                        else:
                            x_batch,y_batch=train_x[iteration*batch_size:(iteration+1)*batch_size,], train_y[iteration*batch_size:(iteration+1)*batch_size]
                            L_training+=1
                        if self.dropout_rate!=0.0:
                            if iteration%10==0:
                                test_loss, summary=sess.run([loss,merged],feed_dict={input_layer:valid_x,output_data:valid_y,is_training_drop:False,is_training_batch:False})
                                test_writer.add_summary(summary, epoch*batch_num+iteration)
                            _,batch_loss, summary=sess.run([self.training_op,loss,merged],feed_dict={input_layer:x_batch,output_data:y_batch,is_training_drop:True,is_training_batch:True})
                            train_writer.add_summary(summary,epoch*batch_num+iteration)
                        else:
                            if iteration%10==0:
                                test_loss, summary=sess.run([loss,merged],feed_dict={input_layer:valid_x,output_data:valid_y,is_training_batch:False})
                                overall_test_loss+=test_loss
                                L_test+=1
                                test_writer.add_summary(summary, epoch*batch_num+iteration)
                            _,batch_loss,summary=sess.run([self.training_op,loss,merged],feed_dict={input_layer:x_batch,output_data:y_batch,is_training_batch:True})
                            train_writer.add_summary(summary,epoch*batch_num+iteration)
                        overall_training_loss+=batch_loss
                   # logging.info()
                   # logger.info("Epoch:%d Finishes, learning rate:%f, Training average loss:%5f,Test average loss:%5f" % (
                     #   epoch + 1, self.learning_rate, 2 * overall_training_loss / (L_training * 187), 2 * overall_test_loss / (187 * L_test)))
                    print("Epoch ",epoch+1, "Finishes","learning rate", self.learning_rate,"Training average loss:", 2*overall_training_loss/(L_training*187),"Test average loss",2*overall_test_loss/(187*L_test))
                self.saver.save(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
                print("The model parameters are saved")

    def train_feedforward_multitask_model(self, train_x, train_y,train_aux,batch_size=256, num_of_epochs=10, shuffle_data=True):
        seed=12345
        np.random.seed(seed)
        print(train_x.shape)
        with self.graph.as_default() as g:
            output_data=tf.placeholder(dtype=tf.float32,shape=(None,self.n_out),name="output_data")
            output_class = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="output_class")
            input_layer=g.get_collection(name="input_layer")[0]
            is_training_batch=g.get_collection(name="is_training_batch")[0]
            if self.dropout_rate!=0.0:
                is_training_drop=g.get_collection(name="is_training_drop")[0]
            with tf.name_scope("loss"):
               output_layer=g.get_collection(name="output_layer_regress")[0]
               output_layer_classification = g.get_collection(name="output_layer_classification")[0]
               loss=tf.reduce_mean(tf.square(output_layer-output_data),name="loss")
               class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer_classification, labels=output_class))
               total_loss = loss + 0.5*class_loss
            with tf.name_scope("train"):
                self.training_op=self.training_op.minimize(total_loss)
            init=tf.global_variables_initializer()
            self.saver=tf.train.Saver()
            with tf.Session() as sess:
                init.run();summary_writer=tf.summary.FileWriter(os.path.join(self.ckpt_dir,"losslog"),sess.graph)
                for epoch in range(num_of_epochs):
                    L=1;overall_loss=0
                    for iteration in range(int(train_x.shape[0]/batch_size)+1):
                        # if iteration == 1374:
                            # pdb.set_trace()
                        if (iteration+1)*batch_size>train_x.shape[0]:
                            x_batch,y_batch,aux_batch=train_x[iteration*batch_size:],train_y[iteration*batch_size:], train_aux[iteration*batch_size:]
                            if list(x_batch)!=[]:
                                L+=1
                            else:continue
                        else:
                            x_batch,y_batch, aux_batch=train_x[iteration*batch_size:(iteration+1)*batch_size,], train_y[iteration*batch_size:(iteration+1)*batch_size], train_aux[iteration*batch_size:(iteration+1)*batch_size]
                            L+=1
                        if self.dropout_rate!=0.0:
                            _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,output_class:aux_batch, is_training_drop:True,is_training_batch:True})
                        else:
                            _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,output_class:aux_batch, is_training_batch:True})
                        overall_loss+=batch_loss
                    print("Epoch ",epoch+1, "Finishes","Training loss:",batch_loss)
                self.saver.save(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
                print("The model parameters are saved")

    def train_feedforward_multitask2_model(self, train_x_c1, train_y_c1,train_x_c2,train_y_c2,batch_size=256, num_of_epochs=10, shuffle_data=True):
        # this is the training for the two different output layers one output layer is for corpus 1 and other is for corpus 2
        seed=12345
        np.random.seed(seed)
        # print(train_x.shape)
        with self.graph.as_default() as g:
            output_data_c1 = tf.placeholder(dtype=tf.float32,shape=(None,self.n_out),name="output_data_c1")
            output_data_c2 = tf.placeholder(dtype=tf.float32, shape=(None, self.n_out), name="output_data_c2")
            input_layer=g.get_collection(name="input_layer")[0]
            is_training_batch=g.get_collection(name="is_training_batch")[0]
            if self.dropout_rate!=0.0:
                is_training_drop=g.get_collection(name="is_training_drop")[0]
            with tf.name_scope("loss"):
               output_layer_c1=g.get_collection(name="output_layer_c1")[0]
               output_layer_c2 = g.get_collection(name="output_layer_c2")[0]
               loss_c1=tf.reduce_mean(tf.square(output_layer_c1-output_data_c1),name="loss_c1")
               loss_c2=tf.reduce_mean(tf.square(output_layer_c2-output_data_c2),name="loss_c2")
            with tf.name_scope("train"):
                    training_op_c1=self.training_op.minimize(loss_c1)
                    training_op_c2=self.training_op.minimize(loss_c2)
            init=tf.global_variables_initializer()
            self.saver=tf.train.Saver()
            batch_num_c1 = int(train_x_c1.shape[0]/batch_size)+1
            batch_num_c2 = int(train_x_c2.shape[0]/batch_size)+1
            with tf.Session() as sess:
                init.run()
                summary_writer=tf.summary.FileWriter(os.path.join(self.ckpt_dir,"losslog"),sess.graph)
                for epoch in range(num_of_epochs):
                    c1_loss = 0
                    c2_loss = 0
                    c1_iter = 0
                    c2_iter = 0
                    for iteration in range(batch_num_c1 + batch_num_c2-1):
                        # choose which to feed 
                        rand_num = random.randint(0,1)
                        # pdb.set_trace()
                        if ((rand_num == 0) & (c1_iter < batch_num_c1)) | ((c2_iter == batch_num_c2) & (c1_iter < batch_num_c1)):
                            # it will choose corpus 1
                            if (c1_iter+1)*batch_size > train_x_c1.shape[0]:
                                x_batch, y_batch = train_x_c1[c1_iter*batch_size:], train_y_c1[c1_iter*batch_size:]
                            else:
                                x_batch,y_batch=train_x_c1[c1_iter*batch_size:(c1_iter+1)*batch_size,], train_y_c1[c1_iter*batch_size:(c1_iter+1)*batch_size] 
                            c1_iter+=1
                            if self.dropout_rate!=0.0:
                                _,batch_loss_c1=sess.run([training_op_c1,loss_c1],feed_dict={input_layer:x_batch,output_data_c1:y_batch,is_training_drop:True,is_training_batch:True})
                            else:
                                _,batch_loss_c1=sess.run([training_op_c1,loss_c1],feed_dict={input_layer:x_batch,output_data_c1:y_batch,is_training_batch:True})
                        # overall_loss_c1+=batch_loss_c1
                        elif ((rand_num == 1) & (c2_iter < batch_num_c2)) | ((c1_iter == batch_num_c1) & (c2_iter < batch_num_c2 )):
                            if (c2_iter+1)*batch_size > train_x_c2.shape[0]:
                                x_batch, y_batch = train_x_c2[c2_iter*batch_size:], train_y_c2[c2_iter*batch_size:]
                            else:
                                x_batch,y_batch=train_x_c2[c2_iter*batch_size:(c2_iter+1)*batch_size,], train_y_c2[c2_iter*batch_size:(c2_iter+1)*batch_size], 
                            c2_iter+=1
                            if self.dropout_rate!=0.0:
                                _,batch_loss_c2=sess.run([training_op_c2,loss_c2],feed_dict={input_layer:x_batch,output_data_c2:y_batch,is_training_drop:True,is_training_batch:True})
                            else:
                                _,batch_loss_c2=sess.run([training_op_c2,loss_c2],feed_dict={input_layer:x_batch,output_data_c2:y_batch,is_training_batch:True})
                        else:
                            continue 
                    print("Epoch ",epoch+1, "Finishes","Training loss c1:",batch_loss_c1, "Training loss c2",batch_loss_c2)
                    print("corpus 1 iterations", c1_iter, "corpus 2 iterations", c2_iter, "batch num c1", batch_num_c1, "batch_num_c2",batch_num_c2)
                self.saver.save(sess,os.path.join(self.ckpt_dir,"multitask_dnn.ckpt"))
                print("The model parameters are saved")


    def train_feedforward_multilingual_model(self, train_x, train_y,train_aux,batch_size=256, num_of_epochs=10, shuffle_data=True):
        seed=12345
        np.random.seed(seed)
        print(train_x.shape)
        with self.graph.as_default() as g:
            f_lang = tf.placeholder(dtype=tf.float32,shape=(None,1024),name="f_lang")
            f_output = tf.placeholder(dtype=tf.float32, shape=(None, self.out),name="f_output")
            s_lang = tf.placeholder(dtype=tf.float32,shape=(None,1024),name="s_lang")
            s_output = tf.placeholder(dtype=tf.float32, shape=(None, self.out),name="s_output")
            # output_class = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="output_class")
            input_layer=g.get_collection(name="input_layer")[0]
            is_training_batch=g.get_collection(name="is_training_batch")[0]
            if self.dropout_rate!=0.0:
                is_training_drop=g.get_collection(name="is_training_drop")[0]
            with tf.name_scope("loss"):
               output_layer=g.get_collection(name="output_layer_regress")[0]
               output_layer_classification = g.get_collection(name="output_layer_classification")[0]
               loss=tf.reduce_mean(tf.square(output_layer-output_data),name="loss")
               class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer_classification, labels=output_class))
               total_loss = loss + 0.5*class_loss
            with tf.name_scope("train"):
                self.training_op=self.training_op.minimize(total_loss)
            init=tf.global_variables_initializer()
            self.saver=tf.train.Saver()
            with tf.Session() as sess:
                init.run();
                summary_writer=tf.summary.FileWriter(os.path.join(self.ckpt_dir,"losslog"),sess.graph)
                for epoch in range(num_of_epochs):
                    L=1;
                    overall_loss=0
                    for iteration in range(int(train_x.shape[0]/batch_size)+1):
                        # if iteration == 1374:
                            # pdb.set_trace()
                        if (iteration+1)*batch_size>train_x.shape[0]:
                            x_batch,y_batch,aux_batch=train_x[iteration*batch_size:],train_y[iteration*batch_size:], train_aux[iteration*batch_size:]
                            if list(x_batch)!=[]:
                                L+=1
                            else:continue
                        else:
                            x_batch,y_batch, aux_batch=train_x[iteration*batch_size:(iteration+1)*batch_size,], train_y[iteration*batch_size:(iteration+1)*batch_size], train_aux[iteration*batch_size:(iteration+1)*batch_size]
                            L+=1
                        if self.dropout_rate!=0.0:
                            _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,output_class:aux_batch, is_training_drop:True,is_training_batch:True})
                        else:
                            _,batch_loss=sess.run([self.training_op,loss],feed_dict={input_layer:x_batch,output_data:y_batch,output_class:aux_batch, is_training_batch:True})
                        overall_loss+=batch_loss
                    print("Epoch ",epoch+1, "Finishes","Training loss:",batch_loss)
                self.saver.save(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
                print("The model parameters are saved")


    def get_batch(self,train_x,train_y,start,batch_size=50):
        utt_keys=list(train_x)
        if (start+1)*batch_size>len(utt_keys):
            batch_keys=utt_keys[start*batch_size:]
        else:
           batch_keys=utt_keys[start*batch_size:(start+1)*batch_size]
        batch_x_dict=dict([(k,train_x[k]) for k  in batch_keys])
        batch_y_dict=dict([(k,train_y[k]) for k in batch_keys])
        utt_len_batch=[len(batch_x_dict[k])for k in list(batch_x_dict)]
        return batch_x_dict, batch_y_dict, utt_len_batch


    def get_valid_batch(self, valid_x, valid_y):
        """
            fetch the validation dataset one time
        """
        utt_keys = list(valid_x)
        valid_x_dict=dict([(k,valid_x[k]) for k  in utt_keys])
        valid_y_dict=dict([(k,valid_y[k]) for k in utt_keys])
        utt_len_valid=[len(valid_x_dict[k])for k in list(valid_x_dict)]
        return valid_x_dict, valid_y_dict, utt_len_valid

    def train_sequence_model(self, train_x, train_y,
                             valid_x, valid_y,
                             batch_size=256, num_of_epochs=10,shuffle_data=False):
        """
            training with sequential model(rnn lstm)
            Args:
                train_x: dictionary format
                dev_x: validation data
        """
        logger = logging.getLogger("train_model")
        logger.info("start the training of seqential model")
        # pdb.set_trace()
        with self.graph.as_default() as g:
            with tf.Session() as sess:
                output_layer = g.get_collection(name="output_layer")[0]
                input_layer = g.get_collection(name="input_layer")[0]
                global_step = g.get_collection(name="global_step")[0]
                learning_rate = g.get_collection(name="learning_rate")[0]
                utt_length_placeholder = g.get_collection(name="utt_length")[0]
                with tf.name_scope("output_data"):
                    output_data = tf.placeholder(tf.float32, shape=(None, None, self.n_out))
                with tf.name_scope("loss"):
                    error = output_data - output_layer
                    loss = tf.reduce_mean(tf.square(error), name="mse_loss")
                tf.summary.scalar('mean_loss', loss)
                tf.summary.scalar('learning_rate', learning_rate)
                merged = tf.summary.merge_all()
                # pdb.set_trace()
                with tf.name_scope("train"):
                    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
                init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
                train_writer = tf.summary.FileWriter(self.ckpt_dir + '/train',
                                      sess.graph)
                valid_writer = tf.summary.FileWriter(self.ckpt_dir + '/test')
                init.run()
                if ckpt:
                    self.saver.restore(sess, os.path.join(self.ckpt_dir, "mymodel.ckpt"))
                x_valid, y_valid, utt_length_valid = self.get_valid_batch(valid_x, valid_y)
                max_length_valid = max(utt_length_valid)
                x_valid = data_utils.transform_data_to_3d_matrix(x_valid, max_length=max_length_valid, shuffle_data=False)
                y_valid = data_utils.transform_data_to_3d_matrix(y_valid, max_length=max_length_valid, shuffle_data=False)
                num_of_batches = len(list(train_x))//batch_size + 1
                logger.info("start training epoches with epoch number {} and batch numbers {}".format(num_of_epochs, num_of_batches))
                for epoch in range(num_of_epochs):
                    for iteration in range(num_of_batches):
                        # pdb.set_trace()
                        # fetch data for each iteration
                        x_batch, y_batch, utt_length_batch=self.get_batch(train_x,train_y,iteration,batch_size)
                        # if the data batch is empty will jump out of the iteration
                        if utt_length_batch==[]:
                            # pdb.set_trace()
                            continue
                        max_length_batch=max(utt_length_batch)
                        x_batch=data_utils.transform_data_to_3d_matrix(x_batch, max_length=max_length_batch, shuffle_data=False)
                        y_batch=data_utils.transform_data_to_3d_matrix(y_batch, max_length=max_length_batch, shuffle_data=False)
                        _, batch_loss,gs, train_summary = sess.run([training_op, loss, global_step, merged],feed_dict={
                            input_layer:x_batch,
                            output_data:y_batch,
                            utt_length_placeholder:utt_length_batch
                        })
                        train_writer.add_summary(train_summary,gs)
                        if gs % 20 == 19:
                            valid_loss, valid_summary = sess.run([loss,merged], feed_dict={
                                input_layer:x_valid,
                                output_data:y_valid,
                                utt_length_placeholder:utt_length_valid
                            })
                            valid_writer.add_summary(valid_summary,gs)
                            logger.info("After epoch {}, training loss is {}, validation loss is {}, global step is {}".format(str(epoch), str(batch_loss),str(valid_loss),str(gs)))
                    self.saver.save(sess, os.path.join(self.ckpt_dir, "mymodel.ckpt"))


    def predict(self, test_x, out_scaler, gen_test_file_list, sequential_training=False, stateful=False):
        """
            predict the results with given model
        """

        io_funcs = BinaryIOCollection()
        test_id_list = list(test_x)
        test_id_list.sort()
        gen_test_file_list.sort()
        test_file_number = len(test_id_list)

        print("generating features on held-out test data...")
        with tf.Session() as sess:
            new_saver=tf.train.import_meta_graph(os.path.join(self.ckpt_dir,"mymodel.ckpt.meta"))
            print("loading the model parameters...")
            output_layer=tf.get_collection("output_layer")[0]
            input_layer=tf.get_collection("input_layer")[0]
            new_saver.restore(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
            print("The model parameters are successfully restored")
            for utt_index in range(test_file_number):
                gen_test_file_name = gen_test_file_list[utt_index]
                temp_test_x        = test_x[test_id_list[utt_index]]
                num_of_rows        = temp_test_x.shape[0]
                if not sequential_training:
                    is_training_batch=tf.get_collection("is_training_batch")[0]
                    if self.dropout_rate!=0.0:
                        is_training_drop=tf.get_collection("is_training_drop")[0]
                        y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,is_training_drop:False,is_training_batch:False})
                    else:
                        y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,is_training_batch:False})
                else:
                    # sequential training
                        temp_test_x=np.reshape(temp_test_x,[1,num_of_rows,self.n_in])
                        hybrid=0
                        utt_length_placeholder=tf.get_collection("utt_length")[0]
                        if "tanh" in self.hidden_layer_type:
                            hybrid=1
                            is_training_batch=tf.get_collection("is_training_batch")[0]
                        if self.dropout_rate!=0.0:
                            is_training_drop=tf.get_collection("is_training_drop")[0]
                            if hybrid:
                                y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:[num_of_rows],is_training_drop:False,is_training_batch:False})
                            else:
                                y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:[num_of_rows],is_training_drop:False})
                        elif hybrid:
                            y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:[num_of_rows],is_training_batch:False})
                        else:
                            y_predict=sess.run(output_layer,feed_dict={input_layer:temp_test_x,utt_length_placeholder:[num_of_rows]})
                data_utils.denorm_data(y_predict, out_scaler)
                io_funcs.array_to_binary_file(y_predict, gen_test_file_name)
                data_utils.drawProgressBar(utt_index+1, test_file_number)
    sys.stdout.write("\n")

class Train_Encoder_Decoder_Models(Encoder_Decoder_Models):
    """
        design a end2end tensorflow model for speech synthesis
    """
    def __init__(self,n_in, hidden_layer_size, n_out, hidden_layer_type, model_dir,output_type='linear',
                 dropout_rate=0.0, loss_function='mse', optimizer='adam',attention=False,cbhg=False):
        Encoder_Decoder_Models.__init__(self,n_in, hidden_layer_size, n_out, hidden_layer_type, output_type='linear',
                                        dropout_rate=0.0, loss_function='mse',
                                        optimizer='adam',attention=attention,cbhg=cbhg)
        self.ckpt_dir=os.path.join(model_dir,"temp_checkpoint_file")

        def get_batch(self,train_x,train_y,start,batch_size):
            utt_keys=list(train_x)
            if (start+1)*batch_size>len(utt_keys):
                batch_keys=utt_keys[start*batch_size:]
            else:
                batch_keys=utt_keys[start*batch_size:(start+1)*batch_size]
            batch_x_dict=dict([(k,train_x[k]) for k  in batch_keys])
            batch_y_dict=dict([(k,train_y[k]) for k in batch_keys])
            utt_len_batch=[len(batch_x_dict[k])for k in list(batch_x_dict)]
            return batch_x_dict, batch_y_dict, utt_len_batch


        def train_encoder_decoder_model(self,train_x,train_y,utt_length,batch_size=1,num_of_epochs=10,shuffle_data=False):
            temp_train_x = data_utils.transform_data_to_3d_matrix(train_x, max_length=self.max_step,shuffle_data=False)
            print("Input shape: "+str(temp_train_x.shape))
            temp_train_y = data_utils.transform_data_to_3d_matrix(train_y, max_length=self.max_step,shuffle_data=False)
            print("Output shape: "+str(temp_train_y.shape))
            with self.graph.as_default() as g:
                outputs=g.get_collection(name="decoder_outputs")[0]
                var=g.get_collection(name="trainable_variables")
                targets=g.get_tensor_by_name("target_sequence/targets:0")
                inputs_data=g.get_tensor_by_name("encoder_input/inputs_data:0")
                if not self.cbhg:
                    inputs_sequence_length=g.get_tensor_by_name("encoder_input/inputs_sequence_length:0")
                target_sequence_length=g.get_tensor_by_name("target_sequence/target_sequence_length:0")
                with tf.name_scope("loss"):
                    error=targets-outputs
                    loss=tf.reduce_mean(tf.square(error))
                gradients=self.training_op.compute_gradients(loss)
                capped_gradients=[(tf.clip_by_value(grad,-5.,5.),var) for grad,var in gradients if grad is not None]
                self.training_op=self.training_op.apply_gradients(capped_gradients)
                init=tf.global_variables_initializer()
                self.saver=tf.train.Saver()
                overall_loss=0;tf.summary.scalar("training_loss",overall_loss)
                with tf.Session() as sess:
                    init.run();tf.summary_writer=tf.summary.FileWriter(os.path.join(self.ckpt_dir,"losslog"),sess.graph)
                    for epoch in range(num_of_epochs):
                        L=1
                        for iteration in range(int(temp_train_x.shape[0]/batch_size)+1):
                            x_batch_dict,y_batch_dict,utt_length_batch=self.get_batch(train_x,train_y,iteration,batch_size)
                            if utt_length_batch==[]:
                                continue
                            else:
                                L+=1
                            assert [len(v) for v in x_batch_dict.values()]==[len(v) for v in y_batch_dict.values()]
                            assert list(x_batch_dict) == list(y_batch_dict)
                            max_length_batch=max(utt_length_batch)
                            x_batch=data_utils.transform_data_to_3d_matrix(x_batch_dict, max_length=max_length_batch, shuffle_data=False)
                            y_batch=data_utils.transform_data_to_3d_matrix(y_batch_dict, max_length=max_length_batch, shuffle_data=False)
                            if self.cbhg:
                                _,batch_loss=sess.run([self.training_op,loss],{inputs_data:x_batch,targets:y_batch,target_sequence_length:utt_length_batch})
                            else:
                                _,batch_loss=sess.run([self.training_op,loss],{inputs_data:x_batch,targets:y_batch,inputs_sequence_length:utt_length_batch,target_sequence_length:utt_length_batch})
                            overall_loss+=batch_loss
                     #if self.cbhg:
                     #    training_loss=loss.eval(feed_dict={inputs_data:temp_train_x,targets:temp_train_y,target_sequence_length:utt_length})
                     #else:
                     #    training_loss=loss.eval(feed_dict={inputs_data:temp_train_x,targets:temp_train_y,inputs_sequence_length:utt_length,target_sequence_length:utt_length})
                        print("Epoch:",epoch+1, "Training loss:",overall_loss/L)
                        summary_writer.add_summary(str(overall_loss),epoch)
                    self.saver.save(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
                    print("The model parameters are saved")

    def predict(self,test_x, out_scaler, gen_test_file_list):
          #### compute predictions ####

        io_funcs = BinaryIOCollection()

        test_id_list = list(test_x)
        test_id_list.sort()
        inference_batch_size=len(test_id_list)
        test_file_number = len(test_id_list)
        with tf.Session(graph=self.graph) as sess:
            new_saver=tf.train.import_meta_graph(self.ckpt_dir,"mymodel.ckpt.meta")
            inputs_data=self.graph.get_collection("inputs_data")[0]
            inputs_sequence_length=self.graph.get_collection("inputs_sequence_length")[0]
            target_sequence_length=self.graph.get_collection("target_sequence_length")[0]
            print("loading the model parameters...")
            new_saver.restore(sess,os.path.join(self.ckpt_dir,"mymodel.ckpt"))
            print("Model parameters are successfully restored")
            print("generating features on held-out test data...")
            for utt_index in range(test_file_number):
                gen_test_file_name = gen_test_file_list[utt_index]
                temp_test_x        = test_x[test_id_list[utt_index]]
                num_of_rows        = temp_test_x.shape[0]

         #utt_length=[len(utt) for utt in test_x.values()]
         #max_step=max(utt_length)
                temp_test_x = tf.reshape(temp_test_x,[1,num_of_rows,self.n_in])

                outputs=np.zeros(shape=[len(test_x),max_step,self.n_out],dtype=np.float32)
                #dec_cell=self.graph.get_collection("decoder_cell")[0]
                print("Generating speech parameters ...")
                for t in range(num_of_rows):
                 #  outputs=sess.run(inference_output,{inputs_data:temp_test_x,inputs_sequence_length:utt_length,\
                #            target_sequence_length:utt_length})
                    _outputs=sess.run(decoder_outputs,feed_dict={inputs_data:temp_test_x,targets:outputs,inputs_sequence_length:[num_of_rows],\
                             target_sequence_length:[num_of_rows]})
                #   #print _outputs[:,t,:]
                    outputs[:,t,:]=_outputs[:,t,:]

                data_utils.denorm_data(outputs, out_scaler)
                io_funcs.array_to_binary_file(outputs, gen_test_file_name)
                data_utils.drawProgressBar(utt_index+1, test_file_number)
        sys.stdout.write("\n")
