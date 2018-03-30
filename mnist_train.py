
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


import minist_inference


# In[3]:


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="/Users/zhouzelun/Documents/python/mnist_data"
MODEL_NAME = "model.ckpt"


# In[4]:


def train(mnist):
    x = tf.placeholder(tf.float32,[None,minist_inference.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,minist_inference.OUTPUT_NODE],name='y-input')
    
    #正则项
    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #计算未使用滑动平均一次前向传播结果
    y = minist_inference.inference(x,regularizer)
    
    #定义当前步数，移动平均时会用到，自动更新+1
    global_step = tf.Variable(0,trainable = False)
    #计算使用滑动平均的前向传播结果
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #在前向传播过后计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    #交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #损失等于交叉商加上正则项
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
    #定义学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,#基础学习速率
                                              global_step,       #当前迭代轮数
                                              500,  #总共需要的迭代次数
                                              LEARNING_RATE_DECAY)      #学习率衰减速率
    #训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate)                    .minimize(loss,global_step = global_step)
    
    train_op = tf.group(train_step,variables_averages_op) 
    #保存模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            
            if i%1000==0:
                print("After %d training step(s),loss on training batch is %g." %(step,loss_value))
                
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)


# In[5]:


def main(argv = None):
    mnist = input_data.read_data_sets(MODEL_SAVE_PATH,one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

