from dataset import Dataset, ground_truth, safe_mkdir

import os
import tensorflow as tf
import numpy as np
from datetime import datetime

class Model:

# All the nodes of the graph are lazily created, meaning they are only created
# when they are called.

    def __init__(self, n_notes, n_steps, n_hidden, learning_rate, device_name="/gpu:0",suffix=""):
        tf.reset_default_graph()
        self.n_notes = n_notes
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.device_name = device_name
        self._prediction = None
        self._pred_thresh = None
        self._cross_entropy = None
        self._cross_entropy2 = None
        self._optimize = None
        self._tp = None
        self._fp = None
        self._fn = None
        self._precision = None
        self._recall = None
        self._f_measure = None
        self.suffix = suffix

    def _transpose_data(self, data):
        return np.transpose(data,[0,2,1])

    def print_params(self):
        print "Learning rate : ",self.learning_rate
        print "Hidden nodes : ",self.n_hidden


# def TP(data,target):
#     return np.sum(np.logical_and(data == 1, target == 1))

    @property
    def tp(self):
        if self._tp is None:
            with tf.device(self.device_name):
                suffix = self.suffix
                pred = self.pred_thresh

                y = tf.get_default_graph().get_tensor_by_name("y"+suffix+":0")
                bool_matrix = tf.logical_and(tf.equal(pred,1),tf.equal(y,1))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._tp = reduced
        return self._tp
    @property
    def fp(self):
        if self._fp is None:
            with tf.device(self.device_name):
                suffix = self.suffix
                pred = self.pred_thresh

                y = tf.get_default_graph().get_tensor_by_name("y"+suffix+":0")
                bool_matrix = tf.logical_and(tf.equal(pred,1),tf.equal(y,0))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._fp = reduced
        return self._fp

    @property
    def fn(self):
        if self._fn is None:
            with tf.device(self.device_name):
                suffix = self.suffix
                pred = self.pred_thresh

                y = tf.get_default_graph().get_tensor_by_name("y"+suffix+":0")
                bool_matrix = tf.logical_and(tf.equal(pred,0),tf.equal(y,1))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._fn = reduced
        return self._fn

    @property
    def precision(self):
        #Returns a vector of length len(dataset), mean has to be computed afterwards
        if self._precision is None:
            with tf.device(self.device_name):
                TP = self.tp
                FP = self.fp
                self._precision = tf.truediv(TP,tf.add(TP,FP))
        return self._precision


    @property
    def recall(self):
        #Returns a vector of length len(dataset), mean has to be computed afterwards
        if self._recall is None:
            with tf.device(self.device_name):
                TP = self.tp
                FN = self.fn
                self._recall = tf.truediv(TP,tf.add(TP,FN))
        return self._recall

    @property
    def f_measure(self):
        #Returns a vector of length len(dataset), mean has to be computed afterwards
        if self._f_measure is None:
            with tf.device(self.device_name):
                prec = self.precision
                rec = self.recall
                self._f_measure = tf.truediv(tf.scalar_mul(2,tf.multiply(prec,rec)),tf.add(prec,rec))
        return self._f_measure

            # def Fmeasure(data,target):
            #     prec = precision(data,target)
            #     rec = recall(data,target)
            #     return 2*prec*rec/(prec+rec)


    @property
    def prediction(self):
        if self._prediction is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_classes = n_notes
                n_steps = self.n_steps
                n_hidden = self.n_hidden
                suffix = self.suffix

                x = tf.placeholder("float", [None,n_steps,n_notes],name="x"+suffix)

                W = tf.Variable(tf.truncated_normal([n_hidden,n_classes]),name="W"+suffix)
                b = tf.Variable(tf.truncated_normal([n_classes]),name="b"+suffix)

                cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True,forget_bias = 1.0)
                outputs, state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32,time_major=False)
                #print(x.get_shape())
                #print(outputs.get_shape())

                outputs = tf.reshape(outputs,[-1,n_hidden])
                pred = tf.matmul(outputs,W) + b
                pred = tf.reshape(pred,[-1,n_steps,n_notes])

                #drop last prediction of each sequence
                pred = pred[:,:n_steps-1,:]

                self._prediction = pred
        return self._prediction

    @property
    def pred_thresh(self):
        if self._pred_thresh is None:
            with tf.device(self.device_name):
                pred = self.prediction
                pred = tf.sigmoid(pred)
                pred = tf.greater(pred,0.5)
                pred = tf.cast(pred,tf.int8)
                self._pred_thresh = pred
        return self._pred_thresh

    @property
    def cross_entropy(self):
        if self._cross_entropy is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_steps = self.n_steps
                suffix = self.suffix
                y = tf.placeholder("float", [None,n_steps-1,n_notes],name="y"+suffix)
                cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=y))
                self._cross_entropy = cross_entropy
        return self._cross_entropy

    @property
    def cross_entropy2(self):
        if self._cross_entropy2 is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_steps = self.n_steps
                suffix = self.suffix
                y = tf.get_default_graph().get_tensor_by_name("y"+suffix+":0")
                cross_entropy2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=y)
                self._cross_entropy2 = cross_entropy2
        return self._cross_entropy2

    @property
    def optimize(self):
        if self._optimize is None:
            with tf.device(self.device_name):
                cross_entropy = self.cross_entropy
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)
                self._optimize = optimizer
        return self._optimize

    def _run_by_batch(self,sess,op,feed_dict,batch_size,mean=True):
        suffix = self.suffix
        x = tf.get_default_graph().get_tensor_by_name("x"+suffix+":0")
        try:
            y = tf.get_default_graph().get_tensor_by_name("y"+suffix+":0")
        except KeyError:
            n_steps = x.get_shape()[1]
            n_notes = x.get_shape()[2]
            y = tf.placeholder("float", [None,n_steps-1,n_notes],name="y"+suffix)

        if y in feed_dict:
            dataset = feed_dict[x]
            target = feed_dict[y]
        else:
            dataset = feed_dict[x]

        no_of_batches = (len(dataset)/batch_size)//1 + 1
        #crosses = np.zeros([dataset.shape[0]])
        #results = np.empty(dataset.shape)
        results = []
        ptr = 0
        for j in range(no_of_batches):
            if y in feed_dict:
                batch_x = dataset[ptr:ptr+batch_size]
                batch_y = target[ptr:ptr+batch_size]
                feed_dict={x: batch_x, y: batch_y}
            else :
                batch_x = dataset[ptr:ptr+batch_size]
                feed_dict={x: batch_x}
            ptr += batch_size
            result_batch = sess.run(op, feed_dict=feed_dict)
            results = np.append(results,result_batch)
        if mean:
            return np.mean(results)
        else :
            return results



    def train(self, data, epochs, batch_size, save_path,display_per_epoch=10,save_step=1,max_to_keep=5,summarize=False):
        # To train a model.
        # data is a dataset object.


        training_data = self._transpose_data(data.train)
        training_target = self._transpose_data(ground_truth(data.train))
        valid_data = self._transpose_data(data.valid)
        valid_target = self._transpose_data(ground_truth(data.valid))
        optimizer = self.optimize
        cross_entropy = self.cross_entropy
        cross_entropy2= self.cross_entropy2
        precision = self.precision
        recall = self.recall
        f_measure = self.f_measure
        suffix = self.suffix

        ckpt_save_path = os.path.join("./ckpt/",save_path)
        summ_save_path = os.path.join("./summ/",save_path)
        safe_mkdir(ckpt_save_path)
        safe_mkdir(summ_save_path,clean=True)



        x = tf.get_default_graph().get_tensor_by_name("x"+suffix+":0")
        y = tf.get_default_graph().get_tensor_by_name("y"+suffix+":0")

        init = tf.global_variables_initializer()
        #launch the graph
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)

        if summarize:
            tf.summary.scalar('cross entropy epoch',cross_entropy,collections=['epoch'])
            tf.summary.scalar('precision epoch',tf.reduce_mean(precision),collections=['epoch'])
            tf.summary.scalar('recall epoch',tf.reduce_mean(recall),collections=['epoch'])
            tf.summary.scalar('f_measure epoch',tf.reduce_mean(f_measure),collections=['epoch'])

            tf.summary.scalar('cross entropy batch',cross_entropy,collections=['batch'])
            tf.summary.scalar('precision batch',tf.reduce_mean(precision),collections=['batch'])
            tf.summary.scalar('recall batch',tf.reduce_mean(recall),collections=['batch'])
            tf.summary.scalar('f_measure batch',tf.reduce_mean(f_measure),collections=['batch'])

            summary_epoch = tf.summary.merge_all('epoch')
            summary_batch = tf.summary.merge_all('batch')
            train_writer = tf.summary.FileWriter(summ_save_path,
                                      sess.graph)


        no_of_batches = (len(training_data)/batch_size)//1 + 1
        display_step = max(int(round(float(no_of_batches)/display_per_epoch)),1)
        n = 0

        saver = tf.train.Saver(max_to_keep=max_to_keep)


        print 'Starting computations : '+str(datetime.now())

        for i in range(epochs):
            ptr = 0
            for j in range(no_of_batches):
                batch_x = training_data[ptr:ptr+batch_size]
                batch_y = training_target[ptr:ptr+batch_size]
                ptr += batch_size
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                if j%display_step == 0 :
                    cross_batch = sess.run(cross_entropy, feed_dict={x: batch_x, y: batch_y})
                    print "Batch "+str(j)+ ", Cross entropy = "+"{:.5f}".format(cross_batch)
                    if summarize:
                        summary_b = sess.run(summary_batch,feed_dict={x: batch_x, y: batch_y})
                        train_writer.add_summary(summary_b,global_step=n)
                        n += 1

            cross = self._run_by_batch(sess,cross_entropy2,{x: valid_data, y: valid_target},batch_size)
            if summarize:
                #summary = self._run_by_batch(sess,summary_op,{x: valid_data, y: valid_target},batch_size)
                summary_e = sess.run(summary_epoch,feed_dict={x: valid_data, y: valid_target})
                train_writer.add_summary(summary_e, global_step=i)
            print "_________________"
            print "Epoch: " + str(i) + ", Cross Entropy = " + \
                          "{:.5f}".format(cross)

            # Save the variables to disk.
            if i%save_step == 0 or i == epochs-1:
                saved = saver.save(sess, os.path.join(ckpt_save_path,"model.ckpt"),global_step=i)
            else :
                saved = saver.save(sess, os.path.join(ckpt_save_path,"model.ckpt"))
            print("Model saved in file: %s" % saved)
            print "_________________"
        print("Optimization finished ! "+str(datetime.now()))
        return

    def load(self,save_path,n_model):
        suffix = self.suffix

        pred = self.prediction
        x = tf.get_default_graph().get_tensor_by_name("x"+suffix+":0")

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver = tf.train.Saver()
        if n_model==None:
            path = tf.train.latest_checkpoint(os.path.join("./ckpt/",save_path))
        else:
            path = os.path.join("./ckpt/",save_path,"model.ckpt-"+str(n_model))
        saver.restore(sess, path)
        return sess, saver

    def run_prediction(self,dataset,save_path,n_model=None,sigmoid=False):
        sess, saver = self.load(save_path,n_model)
        suffix = self.suffix
        pred = self.prediction
        x = tf.get_default_graph().get_tensor_by_name("x"+suffix+":0")

        dataset = self._transpose_data(dataset)

        notes_pred = sess.run(pred, feed_dict = {x: dataset} )
        notes_pred = tf.transpose(notes_pred,[0,2,1])

        if sigmoid:
            notes_pred=tf.sigmoid(notes_pred)

        output = notes_pred.eval(session = sess)
        return output

    def run_prediction2(self,dataset,save_path,n_model=None,batch_size=1,sigmoid=False):
        "TODO : fix behaviour with _run_by_batch"
        sess, saver = self.load(save_path,n_model)
        suffix = self.suffix
        pred = self.prediction
        x = tf.get_default_graph().get_tensor_by_name("x"+suffix+":0")

        dataset = self._transpose_data(dataset)

        notes_pred = self._run_by_batch(sess,pred,{x: dataset},batch_size,mean=False)
        notes_pred = np.transpose(notes_pred,[0,2,1])

        if sigmoid:
            notes_pred=tf.sigmoid(notes_pred)

        output = notes_pred.eval(session = sess)
        return output

    def run_cross_entropy(self,dataset,save_path,n_model=None,batch_size=50):
        sess, saver = self.load(save_path,n_model)
        cross_entropy = self.cross_entropy2

        suffix = self.suffix
        x = tf.get_default_graph().get_tensor_by_name("x"+suffix+":0")
        y = tf.get_default_graph().get_tensor_by_name("y"+suffix+":0")

        target = ground_truth(dataset)
        dataset = self._transpose_data(dataset)
        target = self._transpose_data(target)
        print type(target)

        cross = self._run_by_batch(sess,cross_entropy,{x: dataset,y: target},batch_size)
        return cross



    def compute_eval_metrics_pred(self,dataset,threshold,save_path,batch_size=1,n_model=None):

        preds = self.run_prediction(dataset,save_path,n_model,sigmoid=True)
        idx = preds[:,:,:] > threshold
        preds_thresh = idx.astype(int)

        targets = ground_truth(dataset)

        return self.compute_eval_metrics(preds_thresh,targets,batch_size)

    def compute_eval_metrics(self,data1,data2,batch_size=1,threshold=None):
        if not threshold==None:
            idx = data1[:,:,:] > threshold
            data1 = idx.astype(int)

        prec = precision(data1,data2)
        rec = recall(data1,data2)
        # acc = accuracy(data1,data2)
        F = Fmeasure(data1,data2)
        return F, prec, rec




def TP(data,target):
    return np.sum(np.logical_and(data == 1, target == 1),axis=(1,2))

def FP(data,target):
    return np.sum(np.logical_and(data == 1, target == 0),axis=(1,2))

def FN(data,target):
    return np.sum(np.logical_and(data == 0, target == 1),axis=(1,2))

def precision(data,target,mean=True):

    tp = TP(data,target).astype(float)
    fp = FP(data,target)
    pre_array = tp/(tp+fp+np.full(tp.shape,np.finfo(float).eps))

    if mean:
        return np.mean(pre_array)
    else :
        return pre_array

def recall(data,target,mean=True):
    tp = TP(data,target).astype(float)
    fn = FN(data,target)
    rec_array = tp/(tp+fn+np.full(tp.shape,np.finfo(float).eps))
    if mean:
        return np.mean(rec_array)
    else :
        return rec_array


def accuracy(data,target,mean=True):
    tp = TP(data,target).astype(float)
    fp = FP(data,target)
    fn = FN(data,target)
    acc_array = tp/(tp+fp+fn+np.full(tp.shape,np.finfo(float).eps))
    if mean :
        return np.mean(acc_array)
    else :
        return acc_array

def Fmeasure(data,target,mean=True):
    prec = precision(data,target,mean=False)
    rec = recall(data,target,mean=False)

    if mean:
        return np.mean(2*prec*rec/(prec+rec+np.full(prec.shape,np.finfo(float).eps)))
    else :
        return 2*prec*rec/(prec+rec+np.full(prec.shape,np.finfo(float).eps))

def make_model_from_dataset(dataset,n_hidden,learning_rate,suffix=""):
    n_notes = dataset.get_n_notes()
    n_steps = dataset.get_len_files()
    return Model(n_notes,n_steps,n_hidden,learning_rate,suffix)
