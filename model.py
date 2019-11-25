#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:46:38 2017

@author: hd
"""

import os
import csv
from glob import glob
from datetime import datetime
import time
#from scipy import stats
import re
import numpy as np
import tensorflow as tf
#from utils import save_images
import utils
import scipy.misc
import scipy.io
import slim.ops
import slim.scopes
import slim.losses
import slim.variables
import vgg19

FLAGS = tf.app.flags.FLAGS

BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.9999

class SL(object):
    def __init__(self,input_height=224,input_width=224,batch_size=5):
       self.batch_size = batch_size
       self.input_height = input_height
       self.input_width = input_width
       self.input_fname_pattern = '*.png'
       self.dropout = 0.5

    def cos(vector1,vector2):
        dot_product = 0
        normA = 0.0
        normB = 0.0
        for a,b in zip(vector1,vector2):
            dot_product += a*b
            normA += a**2
            normB += b**2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product/((normA*normB)**0.5)
        
    def calculateSimilarity(self, y1, y2, l):
        y1_n = tf.nn.l2_normalize(y1, 3)
        y2_n = tf.nn.l2_normalize(y2, 3)
        # w = tf.get_variable("linear_weight_%d" % l, [y1.get_shape()[3]])
        # w = tf.abs(w);
        # w = tf.reshape(w, [1, 1, 1, y1.get_shape()[3]])
        # sim_pre = tf.reduce_mean(tf.reduce_sum((y1_n - y2_n)**2 * w, 1), [1, 2])
        # sim_pre = tf.div(1.,1.+tf.reduce_sum(tf.pow(tf.sub(y1,y2),2),1));
        sim_pre = tf.reduce_mean(tf.reduce_sum(y1_n*y2_n, 3), [1, 2])
        return tf.expand_dims(sim_pre, 1)
        
    def losscalculate(self, sim, labels):
        self.similarity = tf.sigmoid(sim)
        loss2 = slim.losses.euclidean_loss(self.similarity, labels, weight=1.0, scope=None)
        loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=sim, labels=labels))
        tf.summary.scalar("loss_crossentropy", loss1)
        tf.summary.scalar("loss_euclidean", loss2)
        return loss1

    def similarity_network(self, features):
        features = tf.concat(features, 1)
        similarity = slim.ops.fc(features, 64, stddev=0.18)
        similarity = slim.ops.fc(similarity, 128, stddev=0.125)
        similarity = slim.ops.fc(similarity, 256, stddev=0.088)
        similarity = slim.ops.fc(similarity, 256, stddev=0.088)
        similarity = slim.ops.fc(similarity, 128, stddev=0.125)
        similarity = slim.ops.fc(similarity, 64, stddev=0.18)
        similarity = slim.ops.fc(similarity, 32, stddev=0.25)
        similarity = slim.ops.fc(similarity, 16, stddev=0.35)
        similarity = slim.ops.fc(similarity, 8, stddev=0.7)
        similarity = slim.ops.fc(similarity, 1, activation=None, stddev=1)
        return similarity
    
    def build_model(self,batch_size):
        
        self.y = tf.placeholder(tf.float32, [self.batch_size,1], name='y')
        
        self.images1 = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, 3], name='images1')
        self.images2 = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, 3], name='images2')
        self.contour1 = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, 3], name='contour1')
        self.contour2 = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, 3], name='contour2')
        
        tf.summary.image('paired_image1', self.images1)
        tf.summary.image('paired_image2', self.images2)
        tf.summary.image('paired_contour1', self.contour1)
        tf.summary.image('paired_contour2', self.contour2)
        
        # self.imagecon1 = tf.concat([self.images1,self.contour1], axis=3)
        # self.imagecon2 = tf.concat([self.images2,self.contour2], axis=3)

        self.images=tf.concat([self.images1, self.contour1, self.images2, self.contour2], axis=0)
        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(self.images)
        features=[vgg.conv1_1, vgg.conv1_2, vgg.conv2_1, vgg.conv2_2, vgg.conv3_1, vgg.conv3_2, vgg.conv3_3, vgg.conv3_4,
                  vgg.conv4_1, vgg.conv4_2, vgg.conv4_3, vgg.conv4_4, vgg.conv5_1, vgg.conv5_2, vgg.conv5_3, vgg.conv5_4]
        isimilarities=[]
        csimilarities=[]
        for l, f in enumerate(features):
            b1, b2 = tf.split(f, 2, 0)
            if1, cf1 = tf.split(b1, 2, 0)
            if2, cf2 = tf.split(b2, 2, 0)
            isimilarities.append(self.calculateSimilarity(if1, if2, l))
            csimilarities.append(self.calculateSimilarity(cf1, cf2, l))
        similarities = isimilarities + csimilarities    

        self.similarity_logits=self.similarity_network(similarities)
        self.loss = self.losscalculate(self.similarity_logits, self.y)
        
    def train(self):
        data = glob(os.path.join(
          "data", FLAGS.dataset, self.input_fname_pattern))
        datacontourdir = os.path.join("data", FLAGS.datasetcontour)
        # c=input()
        csvfile=open('sim_gro_mer.csv','r')
        #csvfile=open('ISOMap8Similarity.csv','r')
        si = np.array([[float(e) for e in l] for l in csv.reader(csvfile)])
        csvfile.close()
        imagepairs=[]
        filepairscon=[]
        labels=[]
        for i in range(400):
            for j in range(i, 400):
                imagepairs.append([data[i], data[j]])
                m = int(re.split('[\\\\\\-/]', data[i])[-2])
                n = int(re.split('[\\\\\\-/]', data[j])[-2])
                filepairscon.append([os.path.join(datacontourdir, '%d-Edge.png' % m),os.path.join(datacontourdir, '%d-Edge.png' % n)])
                #print("({0},{1}),({2},{3})".format(data[i],data[j],os.path.join(datacontourdir, '%d-Edge.png' % m),os.path.join(datacontourdir, '%d-Edge.png' % n)))
                #print("(%d,%d)" % (m, n))
                labels.append(si[m-1][n-1])
                #print("(%d,%d)" % (m, n))
                #print((si[m-1][n-1]))
                
        shuffled_index = np.arange(len(imagepairs))
        np.random.seed(12345)
        np.random.shuffle(shuffled_index)
        imagepairs = np.array(imagepairs)
        imagepairs = imagepairs[shuffled_index]
        filepairscon = np.array(filepairscon)
        filepairscon = filepairscon[shuffled_index]
        labels = np.array(labels)
        labels = labels[shuffled_index]
        
        imagepairs = imagepairs[0:80200]
        filepairscon = filepairscon[0:80200]
        
        labels = labels[0:80200]
        num_batches = len(imagepairs)//self.batch_size
        with tf.device('/gpu:0'):
            # batch_norm_params = {'decay': BATCHNORM_MOVING_AVERAGE_DECAY,'epsilon': 1e-5}
            # Set weight_decay for weights in Conv and FC layers.
            with slim.scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], 
                                stddev=0.02, 
                                activation=tf.nn.relu, 
                                batch_norm_params=None,
                                weight_decay=0):
                self.build_model(FLAGS.batch_size)
                
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(self.loss)

                          
    
#            batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
    
        # Add a summaries for the input processing and global_step.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        # Group all updates to into a single train op.
#        batchnorm_updates_op = tf.group(*batchnorm_updates)
#        train_op = tf.group(train_op, batchnorm_updates_op)
  
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
  
        summary_op = tf.summary.merge(summaries)
  
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
  
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
  
  
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, ckpt.model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
              (datetime.now(), FLAGS.checkpoint_dir))
  
        summary_writer = tf.summary.FileWriter(
          FLAGS.log_dir,
          graph=sess.graph) 
  
        for step in range(FLAGS.max_steps):
            idx = step % num_batches
            if idx==0:
                shuffled_index = np.arange(80200)
                #shuffled_index = np.arange(45150)
                np.random.shuffle(shuffled_index)
                imagepairs = imagepairs[shuffled_index]
                filepairscon = filepairscon[shuffled_index]
                labels = labels[shuffled_index]

            batch_files = imagepairs[idx*self.batch_size:(idx+1)*self.batch_size]
            batch_files1, batch_files2 = zip(*batch_files)
            batch_filescon = filepairscon[idx*self.batch_size:(idx+1)*self.batch_size]
            batch_filescon1, batch_filescon2 = zip(*batch_filescon)

            batch1 = np.array([utils.load_image(batch_file) for batch_file in batch_files1])
            batch2 = np.array([utils.load_image(batch_file) for batch_file in batch_files2])
            
            batchcon1 = np.array([utils.load_image(batch_filescon) for batch_filescon in batch_filescon1])
            batchcon2 = np.array([utils.load_image(batch_filescon) for batch_filescon in batch_filescon2])
            
            batch_labels = np.array(labels[idx*self.batch_size:(idx+1)*self.batch_size])
            #print((batch1))
            #print((batch2))
            
            batch1 = np.expand_dims(batch1, 3)
            batch1 = np.tile(batch1, (1, 1, 1, 3))
            batch2 = np.expand_dims(batch2, 3)
            batch2 = np.tile(batch2, (1, 1, 1, 3))
            batchcon1 = np.expand_dims(batchcon1, 3)
            batchcon2 = np.expand_dims(batchcon2, 3)
            batchcon1 = np.tile(batchcon1, (1, 1, 1, 3))
            batchcon2 = np.tile(batchcon2, (1, 1, 1, 3))
            
            batch_labels = np.expand_dims(batch_labels, 1)
            
            start_time = time.time()
            sess.run([train_op], feed_dict={self.images1: batch1, self.images2: batch2,self.contour1: batchcon1, self.contour2: batchcon2, self.y: batch_labels})
           # sess.run([train_op], feed_dict={self.images1: batch1, self.images2: batch2, self.y: batch_labels})
            duration = time.time() - start_time
  
            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d(%.1f examples/sec; %.3f '
                            'sec/batch)')
                print(format_str % (datetime.now(), step, examples_per_sec, duration))
      
            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={self.images1: batch1, self.images2: batch2,self.contour1: batchcon1, self.contour2: batchcon2, self.y: batch_labels})
                summary_writer.add_summary(summary_str, step)
        
            # Save the model checkpoint periodically.
            if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        
    def visualize(self):
        with tf.device('/gpu:0'):
            # batch_norm_params = {'decay': BATCHNORM_MOVING_AVERAGE_DECAY,'epsilon': 1e-5}
            # Set weight_decay for weights in Conv and FC layers.
            with slim.scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], 
                                stddev=0.02, 
                                activation=tf.nn.relu, 
                                batch_norm_params=None,
                                weight_decay=0):
                self.build_model(FLAGS.batch_size)
                
        sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement))


        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, ckpt.model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                (datetime.now(), FLAGS.checkpoint_dir))
        w = slim.variables.get_variables_by_name('weights', 'FC')
        w1_eval=sess.run(w[0])
        scipy.io.savemat('w1.mat',{'w1': w1_eval})
        w1_eval=np.abs(w1_eval)
        w1_eval=w1_eval/np.max(w1_eval)
        #w1_eval=(w1_eval-np.min(w1_eval))/(np.max(w1_eval)-np.min(w1_eval))
        w1_eval=np.kron(w1_eval, np.ones((13,13)))
        scipy.misc.imsave('w1.png', w1_eval)
