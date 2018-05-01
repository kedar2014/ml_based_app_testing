import tensorflow as tf
import numpy as np
import os
import glob
import re
import util_classifier as ut
import models as model

#tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

logs_path = './logs/1'
image_folder_path = './temp'


#Generate batch parameters
batch_size = 50
num_preprocess_threads = 1
min_queue_examples = 256
resume = False
util_obj = ut.Utilities()


# Model Hyper parameters
lr_rate = 0.001
decay_rate = 0.01
num_epochs=5000

#grid_tf = tf.placeholder(tf.float32,shape=[3, 0, 1, 2],name = 'grid_tf')

master_labels_arr = np.array([])
master_labels_arr = [name for name in os.listdir(image_folder_path) if os.path.isdir(image_folder_path+'/'+name)]
categories = len(master_labels_arr)
master_labels = tf.convert_to_tensor(master_labels_arr)

def get_data(path,master_labels_arr):
    """
    Return image_paths, labels such that label[i] corresponds to image_paths[i].

    image_paths: list of strings
    labels: list/np array of labels
    """

    image_paths = glob.glob(path + '/**/*.jpg', recursive=True)
    labels = []
    labels = [ master_labels_arr.index(re.search(".*/(.*).jpg", image_path).group(1)[:-5]) for image_path in image_paths ]

    return  image_paths, labels  

def preprocess_image_tensor(image_tf):
    """Preprocess a single image."""
    image = tf.image.convert_image_dtype(image_tf, dtype=tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(image, 250, 250)
    #image = tf.image.per_image_standardization(image)
    return image


image_paths, labels  = get_data(image_folder_path,master_labels_arr)
image_paths_tf = tf.convert_to_tensor(image_paths, dtype=tf.string, name='image_paths')
labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32, name='labels')
image_path_tf, label_tf = tf.train.slice_input_producer([image_paths_tf, labels_tf], shuffle=True)
label_one_hot = tf.one_hot(label_tf,categories,1.0,0.0, name="one_hot_label")

image_buffer_tf = tf.read_file(image_path_tf, name='image_buffer')
image_tf = tf.image.decode_jpeg(image_buffer_tf, channels=3, name='image')
image_tf = preprocess_image_tensor(image_tf)

# Generate batch
labels,images = tf.train.shuffle_batch([label_tf,image_tf],batch_size=batch_size,num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)
print_op = tf.Print(labels,[labels])

#first set of CNN -> RELU -> MAX Pool -> Norm
cnn_set_1 = model.cnn_layer_set(images,'CNN_Layer_Set1',[3,3,3,32])

#get the weights of first conv layer and put them on the tensorboard
with tf.variable_scope('CNN_Layer_Set1/variables'):
    filter = tf.get_variable(name='weights',shape=filter,initializer=tf.truncated_normal_initializer())

#Second set of CNN -> RELU -> MAX Pool -> Norm
cnn_set_2 = model.cnn_layer_set(cnn_set_1,'CNN_Layer_Set2',[3,3,32,64])

#Third set of CNN -> RELU -> MAX Pool -> Norm
cnn_set_3 = model.cnn_layer_set(cnn_set_2,'CNN_Layer_Set3',[3,3,64,64])

#Four set of CNN -> RELU -> MAX Pool -> Norm
cnn_set_4 = model.cnn_layer_set(cnn_set_3,'CNN_Layer_Set4',[3,3,64,64])

pool2_flat = tf.reshape(cnn_set_4, [-1, 16 * 16 * 64])

with tf.name_scope('FC1'):
    fc1 = tf.layers.dense(pool2_flat,1024,activation=tf.nn.relu,kernel_initializer=None,name="FC1")
with tf.name_scope('FC2'):    
    output = tf.layers.dense(fc1,categories,activation=None,kernel_initializer=None,name="FC2")



with tf.name_scope('Training'):
    cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=labels)
    loss = tf.reduce_sum(cross_entropies)
    

    tf.summary.scalar('Loss',loss)
    
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate,decay=decay_rate,momentum=0.3,epsilon=1e-10, name = 'RMSProp')
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_rate,name = 'GradientDescent')
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate,beta1=0.9,beta2=0.999,epsilon=1e-8,name="Adam")

    grads_and_vars = optimizer.compute_gradients(loss,var_list=tf.trainable_variables())
    train_op = optimizer.apply_gradients(grads_and_vars)

with tf.name_scope('Top_K_Predictions'):
    top_k_op = tf.nn.in_top_k(output, labels, 1)
    #accuracy_op = tf.summary.scalar('Accuracy',accuracy_model)


init = tf.global_variables_initializer()
#image_op = tf.summary.image('conv1/kernels', grid, max_outputs=1)
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter(logs_path,graph=sess.graph)
    saver = tf.train.Saver(var_list=tf.trainable_variables()) 
    if resume:
        saver.restore(sess,logs_path + '/image_classifier.ckpt')
        

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord=coord)

    try:
        for step in range(num_epochs):
            
            # with tf.train.MonitoredTrainingSession(master='',is_chief=True,save_checkpoint_secs=60,checkpoint_dir=logs_path) as sess:
                _ , loss_value,s,correct_predictions = sess.run([train_op,loss,merged_summary,top_k_op])
                accuracy,total_correct = util_obj.calc_accuracy(correct_predictions,(step+1)*batch_size)
                print('Step:-' + str(step) + ", Accuracy- " + str(accuracy) + "% - correct for batch:-" + str(total_correct) )
                #sess.run(accuracy_op,feed_dict=accuracy)
       
                #sess.run(print_op)
                writer.add_summary(s,global_step=step)
                writer.flush()
                saver = tf.train.Saver(var_list=tf.trainable_variables()) 
                saver.save(sess, logs_path + '/image_classifier.ckpt', global_step=None,write_meta_graph=True)
                
    except Exception as e:
            print(str(e))
            coord.request_stop()
    finally:
            coord.request_stop()
            coord.join(threads)