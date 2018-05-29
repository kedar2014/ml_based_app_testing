#!/bin/sh

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import app_facing_code as app_code
import os

H1 = 32 # number of hidden layer neurons
H2 = 64
H3 = 128
batch_size = 1 # every how many episodes to do a param update?
lr_rate = 0.001
gamma = 0.90 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
train = True
episode_number = 0
logs_path = './logs/QBR/'

target_machine = "pc" if os.environ['TARGET_MACHINE'] == None else os.environ['TARGET_MACHINE']
env = app_code.AppFacing(target_machine,render)

observation = env.reset()
x,y = env.get_observation_size()
D = x * y


input_x = tf.placeholder(shape=[None,D], dtype=tf.float32)
actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")
py_labels = []
observations_input = []
reward_list_discounted = []
total_rewards = 0
reward_list = []

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  running_add = 0
  discounted_r = np.zeros_like(r,dtype=float)
  for t in reversed(range(0, len(r))):
    running_add = running_add *(gamma**(len(r)-1-t)) + r[t]
    discounted_r[t] = running_add


  return discounted_r                            

 #variable tensors

#if not resume:
with tf.name_scope('Model'):
    input_nn = tf.convert_to_tensor(input_x)
    #W1 = tf.get_variable(name="W1",shape=H,dtype=tf.float32)
    #W3 = tf.constant_initializer(W1)
    ly1 = tf.layers.dense(input_nn,H1,
                            use_bias=False,
                            kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1),
                            activation=tf.nn.relu,
                            name="first_Layer",
                            trainable=True)
    tf.summary.histogram("Weights_FirstLayer",tf.trainable_variables()[0])
    
    # img1 = tf.reshape(tf.transpose(tf.trainable_variables()[0],None),[-1,y,x,1])
    # tf.summary.image('weight_layer1',img1,max_outputs=50)                        

    ly2 = tf.layers.dense(ly1,H2,
                            use_bias=False,
                            kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1),
                            activation=tf.nn.relu,
                            name="Second_hidden",
                            trainable=True)

    ly3 = tf.layers.dense(ly2,H3,
                            use_bias=False,
                            kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01),
                            activation=tf.nn.relu,
                            name="Third_hidden",
                            trainable=True)                        
                     
    output = tf.layers.dense(ly3,6,use_bias=False,
                            kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01),
                            activation=None,
                            name="Output_Layer",
                            trainable=True)
    tf.summary.histogram("Weights_SecondLayer",tf.trainable_variables()[1])    
    softmax_op = tf.nn.softmax(output)
    action_op = tf.multinomial(logits = output,num_samples=1,name='action_sampler')

with tf.name_scope('Training'):
    #cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions,logits=output)
    cross_entropies = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions,6),logits=output,reduction=tf.losses.Reduction.NONE)
    loss_pre = rewards * cross_entropies
    loss = tf.reduce_sum(loss_pre)
   
    tf.summary.scalar('Loss',loss)
    
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate,decay=decay_rate,momentum=0.3,epsilon=1e-10, name = 'RMSProp')

    grads_and_vars = optimizer.compute_gradients(loss,var_list=tf.trainable_variables())
    train_op = optimizer.apply_gradients(grads_and_vars)

global_step = tf.Variable(0, name='global_step', trainable=False)
init = tf.global_variables_initializer()
merged_summary = tf.summary.merge_all()


with tf.Session() as sess:
    
    sess.run(init)
    writer = tf.summary.FileWriter(logs_path,graph=sess.graph)
    saver = tf.train.Saver(tf.trainable_variables(),{"global_step":global_step})
    
    if resume:
        saver.restore(sess,logs_path + 'treasure.ckpt')
        episode_number = global_step.eval()
  
    while True:
          
        action = sess.run(action_op, feed_dict={input_x : [observation]})
        observations_input.append(observation)
        py_labels.append(action[0][0])

        observation, reward, done, info = env.step(action[0][0])   # 6 possible actions
       
        total_rewards += reward

        reward_list.append(reward)
        
        if done:
                        
            episode_number+=1
            print ('ep %d: game finished, total rewards: %f, actions taken %d -> %d' % (episode_number, total_rewards, py_labels[0],py_labels[1])) 

            reward_list_discounted = np.hstack((reward_list_discounted, discount_rewards(reward_list)))
            reward_list = []
            total_rewards = 0
            observation = env.reset()
            if episode_number % batch_size == 0 and train :

                _, s,loss_val = sess.run([train_op,merged_summary,loss],feed_dict={input_x : observations_input,actions: py_labels,rewards: reward_list_discounted})
                print ('episode number :- ', episode_number)
                print ('loss    ', loss_val)
                                
                writer.add_summary(s,global_step=episode_number)
                writer.flush()
                global_step_op = tf.assign(global_step, episode_number)
                sess.run(global_step_op)
                saver.save(sess, logs_path + 'treasure.ckpt', global_step=None,write_meta_graph=True)
                reward_list,py_labels,observations_input,reward_list_discounted = [],[],[],[]

        









