import tensorflow as tf
from data_loader import *
from abc import abstractmethod
import math
import numpy as np
import random
from numpy import array
import sys

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size, 
                val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes)
        self.num_epochs=num_epochs
        self.train_paths = glob.glob(os.path.join(train_images_dir, "**/*.png"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_images_dir, "**/*.png"), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_images_dir, "**/*.png"), recursive=True)
        self.base_dir=base_dir

        self.train_batch_size=train_batch_size
        self. val_batch_size= val_batch_size
        self. test_batch_size= test_batch_size

        self.learning_rate=learning_rate
        self.width_of_image=width_of_image
        self.height_of_image=height_of_image
        self.num_classes=num_classes
        self.max_to_keep=max_to_keep
        self.model_name=model_name
    def get_optimizer(self, learning_rate):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        return train_op

    def create_network(self):
        # tf.reset_default_graph()
     
        self.x = tf.placeholder("float",[None,self.height_of_image,self.width_of_image] ,name="x")
        self.y = tf.placeholder("float",[None,self.num_classes],  name="y")
    
        self.prediction=self.network(self.x)

        self.loss = self.metrics(self.y,self.prediction)[0]
        self.accuracy=self.metrics(self.y,self.prediction)[1]
        self.optim=self.get_optimizer(self.learning_rate)
    #     self.optim = tf.train.GradientDescentOptimizer(
    # learning_rate=self.learning_rate).minimize(self.loss)

        # self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        
    def load(self):
        print(" [*] Reading checkpoint...")
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        checkpoint_dir = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(ckpt_name)
            return True
        else:
            return False

   

    def initialize_network(self):
        self.sess= tf.InteractiveSession()
        if os.path.exists(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
            self.sess.run(tf.global_variables_initializer())
           
        else:
            self.load()
           
        

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        num_complete_minibatches = len(self.train_paths)// self.train_batch_size
        num_complete_minibatches_val = len(self.val_paths)// self.val_batch_size
        
        
        x_val=[]
        y_val=[]
        for i in range(num_complete_minibatches_val):
            x_val1,y_val1=self.data_loader.val_data_loader(i)
            x_val=x_val+x_val1
            y_val=y_val+y_val1
        


        for epoch in range(self.num_epochs):
            random.shuffle(self.data_loader.train_paths)
            mini_batches=[]
            for k in range(num_complete_minibatches):
                mini_batches.append(self.data_loader.train_data_loader(k))
            
            epoch_cost = 0
            epoch_acc = 0
            
            for minibatch in mini_batches:
                (minibatch_x, minibatch_y) = minibatch
                
                minibatch_optimizer, minibatch_cost, minibatch_prediction, minibatch_acc  = self.sess.run([self.optim, self.loss, self.prediction, self.accuracy], feed_dict = {self.x: minibatch_x, self.y: minibatch_y})
                
                epoch_cost += minibatch_cost
                epoch_acc += minibatch_acc

            epoch_cost = epoch_cost/num_complete_minibatches
            epoch_acc = epoch_acc/num_complete_minibatches

            
            if epoch%display_step ==0:
                print("cost after epoch %i :  %.3f" % (epoch + 1, epoch_cost), end="")
                print("  train accuracy   :  %.3f" % epoch_acc)
                

            if epoch%validation_step ==0:
                
                val_optimizer, val_loss, val_prediction, val_accuracy  = self.sess.run([self.optim, self.loss, self.prediction, self.accuracy], feed_dict = {self.x: x_val, self.y: y_val})
                print("  val accuracy   :  %.3f" % (val_accuracy))
               
                

            if epoch%checkpoint_step ==0:
                self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
                    
                    os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'),exist_ok=True)
                    self.saver.save(self.sess, os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints','my-model'))
                    #,global_step=1
                else:
                    self.saver.save(self.sess, os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints','my-model'))


            if epoch%summary_step ==0:
               
                if os.path.isfile(os.path.join(os.getcwd(),self.base_dir,self.model_name, 'summaries'))== False:
                    os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'),exist_ok=True)
                    tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
                else:
                    tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)

          
        print("network trained")
        

    def test_model(self):
        # self.load()
        num_complete_minibatches_test = len(self.test_paths) // self.test_batch_size
        x_test = []
        y_test = []
        for i in range(num_complete_minibatches_test):
            x_test1, y_test1 = self.data_loader.test_data_loader(i)
            x_test = x_test + x_test1
            y_test = y_test + y_test1
        test_accuracy = self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test})
        print("  test accuracy   :  %.3f" % (test_accuracy))





    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')


