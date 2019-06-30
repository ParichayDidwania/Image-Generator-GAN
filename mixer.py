import tensorflow as tf
from reading_tf import read_data
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

def gen(data, reuse=False):


    conv1_fil = tf.Variable(tf.truncated_normal([4,4,3,128],mean = 0, stddev = 0.08))
    conv2_fil = tf.Variable(tf.truncated_normal([4,4,128,128],mean = 0, stddev = 0.08))
    conv3_fil = tf.Variable(tf.truncated_normal([4,4,128,3],mean = 0, stddev = 0.08))

    hidden1=tf.layers.dense(inputs=data,units=2352,activation=tf.nn.leaky_relu)

    reshape = tf.reshape(hidden1,[1,28,28,3])

    conv1 = tf.nn.conv2d(reshape,conv1_fil,[1,1,1,1],padding='SAME')
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.leaky_relu(conv1)

    conv2 = tf.nn.conv2d(conv1,conv2_fil,[1,1,1,1],padding='SAME')
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.leaky_relu(conv2)

    conv3 = tf.nn.conv2d(conv2,conv2_fil,[1,1,1,1],padding='SAME')
    conv3 = tf.layers.batch_normalization(conv3)
    conv3 = tf.nn.leaky_relu(conv3)

    conv4 = tf.nn.conv2d(conv3,conv3_fil,[1,1,1,1],padding='SAME')
    conv4 = tf.nn.sigmoid(conv4)

    return conv4
    
def discriminator(data,reuse=False):
    
    conv1_fil = tf.Variable(tf.random_normal([4,4,3,128]))
    conv2_fil = tf.Variable(tf.random_normal([4,4,128,128]))
    conv3_fil = tf.Variable(tf.random_normal([4,4,128,64]))

    conv1 = tf.nn.conv2d(data,conv1_fil,[1,1,1,1],padding='SAME')
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.leaky_relu(conv1)

    conv2 = tf.nn.conv2d(conv1,conv2_fil,[1,1,1,1],padding='SAME')
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.leaky_relu(conv2)

    conv3 = tf.nn.conv2d(conv2,conv2_fil,[1,1,1,1],padding='SAME')
    conv3 = tf.nn.leaky_relu(conv3)

    conv4 = tf.nn.conv2d(conv3,conv2_fil,[1,1,1,1],padding='SAME')
    conv4 = tf.nn.leaky_relu(conv4)
    
    conv5 = tf.nn.conv2d(conv2,conv3_fil,[1,1,1,1],padding='SAME')
    conv5 = tf.nn.sigmoid(conv5)

    return conv5
   

def train(data,epoch=3):

    z = tf.placeholder(tf.float32,[None,100],name='z1')

    iterator = tf.data.Iterator.from_structure(output_types = data.output_types, output_shapes=data.output_shapes)
    real = iterator.get_next()

    train_op = iterator.make_initializer(data,name='train_op')
    
    G_sample = gen(z)
    r_logit = discriminator(real)
    f_logit = discriminator(G_sample,reuse=True)
   
    with tf.name_scope('loss'):

        loss_dis1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logit,labels=tf.ones_like(r_logit))) 
        loss_dis2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit,labels=tf.zeros_like(f_logit)))

        loss_dis = loss_dis1 + loss_dis2

        loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logit,labels =tf.ones_like(f_logit)))

        loss_pixel_wise = tf.reduce_mean(abs(G_sample-real))

        gen_optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss_gen)
        dis_optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss_dis)
        pixel_optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss_pixel_wise)

        tf.summary.scalar('loss_discriminator',loss_dis)
        tf.summary.scalar('loss_generator',loss_gen)
        tf.summary.scalar('loss_pix2pix',loss_pixel_wise)
        merge = tf.summary.merge_all()

        saver = tf.train.Saver()
    
    init_op = tf.global_variables_initializer()
    j=0
    
    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('./graph/',sess.graph)
        for i in range(epoch):
            sess.run(train_op)    
            while(True):
                try:
                    z_batch = np.random.uniform(-1, 1,[1,100])

                    '''
                    To Display the Input images

                    l=[0]
                    if(j in l):
                        z = sess.run(real)
                        print("A")                        
                        z = z*255
                        z = z.astype(np.uint8)                        
                        img = Image.fromarray(z[-1,:,:,:], 'RGB')
                        img.show()
                    ''' 
                    # Display generated images
                    if(j%1000==0):
                        img = sess.run(G_sample,feed_dict={z:z_batch})
                        img = img*255
                        img = img.astype(np.uint8)
                        img = Image.fromarray(img[-1,:,:,:], 'RGB')
                        img.show()
                    

                    if(j%100==0):
                        l_dis,_ = sess.run([loss_dis,dis_optimizer],feed_dict={z:z_batch})
                        l_gen,_ = sess.run([loss_gen,gen_optimizer],feed_dict={z:z_batch})
                        l_pixel,_ = sess.run([loss_pixel_wise,pixel_optimizer],feed_dict={z:z_batch})
                        print("loss of dis : ",l_dis," loss of gen : ",l_gen," loss of pixel : ",l_pixel," AT step : ",j )
                    else:
                        summ,_1,_2,_3 = sess.run([merge,dis_optimizer,gen_optimizer,pixel_optimizer],feed_dict={z:z_batch})
                        writer.add_summary(summ,j)                   
                    j+=1
                    
                except tf.errors.OutOfRangeError:
                    
                    break
            
        saver.save(sess,'./model/mix')

path = './mix.tfrecords'
dataset = read_data(path)
train(dataset)


