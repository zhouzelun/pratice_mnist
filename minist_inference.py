
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


INPUT_NODE = 784 #输入层的节点数。对于MNIST数据集，这个就等于图片的像素。
OUTPUT_NODE = 10 #输出层的节点数。这个等于类别的数目。因为在MNIST数据集中需要区分的事0-9，所以这里输出层的节点数为10。
LAYER1_NODE = 500 #隐藏层节点数


# In[4]:


def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer = tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights


# In[5]:


def inference(input_tensor,regularizer):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    #这里实际含义是： avg_class == None 时，是训练时的前向传播过程，else时是为了在测试时计算准确里用的
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable(name = 'biases',shape=[LAYER1_NODE],                                    initializer = tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable(name = 'biases',shape=[OUTPUT_NODE],                                    initializer = tf.constant_initializer(0.0))
        return tf.matmul(layer1,weights)+biases
    


# In[6]:


'''def train():
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    #truncated_normal生成正太分布值
    #隐藏层参数
    #weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    #biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #输出层参数
    #weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    #biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    #计算未使用滑动平均一次前向传播结果
    y = inference(x,None)
    #定义当前步数，移动平均时会用到，自动更新+1
    global_step = tf.Variable(0,trainable = False)
    #计算使用滑动平均的前向传播结果
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x,variable_averages,True)
    test_y = inference(x,None,True)
    #在前向传播过后计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_y,labels=tf.argmax(y_,1))
    #交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #正则项
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    with tf.variable_scope("",reuse = True):
        regularization = regularizer(tf.get_variable("layer1/weights"))+regularizer(tf.get_variable("layer2/weights"))
    #损失等于交叉商加上正则项
    loss = cross_entropy_mean + regularization
    #定义学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,#基础学习速率
                                              global_step,       #当前迭代轮数
                                              500,  #总共需要的迭代次数
                                              LEARNING_RATE_DECAY)      #学习率衰减速率
    #训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(loss,global_step = global_step)
    #反向传播和滑动平均更新参数，这里直接实现了前向及逆向传播过程，在利用滑动平均更新参数的一整个过程
    #with tf.control_dependencies([train_step,variables_averages_op]):
        #train_op = tf.no_op(name='train')
    train_op = tf.group(train_step,variables_averages_op) 
    
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #计算出准确度，此处将bool转换成0，1，再用reduce_mean算1占的比例就可以得出准确度，可用一下注释代码验证
    #tmp = tf.Variable([True,False,True])
    #tmp1  = tf.cast(tmp,dtype=tf.float32)
    #with tf.Session() as sess1:
        #tf.global_variables_initializer().run()
        #print(sess1.run(tf.reduce_mean(sess1.run(tmp1))))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        ######################
        data = sc.loadmat('loc/last.mat')
        result=pd.DataFrame(data['last'])
        X = {}
        for i in range(23):
            X[i] = result[result[180] == i]
        X[1][181] = 45
        X[2][181] = 33
        X[3][181] = 21
        X[12][181] = 9
        
        test = pd.concat([X[1],X[2],X[3]])
        y = test[181]
        x = test.drop([180,181],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        x1 = tf.Variable(tf.constant(X_train))
        x2 = tf.Variable(tf.constant(X_test))
        y1 = tf.Variable(tf.constant(y_train))
        y2 = tf.Variable(tf.constant(y_test))
        ######################
        #验证数据
        tf.global_variables_initializer().run()
        validate_feed = {x:y1,y_:y2}
        #测试数据
        test_feed = {x:y1,y_:y2}
    
        for i in range(TRAINING_STEPS):
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training step(s),validation accuracy " "using average model is %g" %(i,validate_acc))
            sess.run(train_op,feed_dict={x:y1,y_:y2})
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("after %d training step(s),validation accuracy " "using average model is %g" %(TRAINING_STEPS,test_acc))
        print()'''

