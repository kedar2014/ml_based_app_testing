import tensorflow as tf





def cnn_layer_set(input,cnn_name_space,filter):

   
     with tf.name_scope(cnn_name_space +'/Conv_Layer'):
      with tf.variable_scope(cnn_name_space + '/variables'):
       filter = tf.get_variable(name='weights',shape=filter,initializer=tf.truncated_normal_initializer())
      cnn = tf.nn.conv2d(input,filter,[1,1,1,1],padding='SAME')
    
     with tf.name_scope(cnn_name_space + '/Relu'):
      relu = tf.nn.relu(cnn,name="Relu")

     with tf.name_scope(cnn_name_space + '/MaxPool'):    
        pool = tf.nn.max_pool(relu,[1,3,3,1],[1,2,2,1],padding='SAME',name='Pool')

     with tf.name_scope(cnn_name_space + '/Norm'):     
        output = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='Norm')

     return output   