import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

z_batch = np.random.uniform(-1, 1,[1,100])

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model/mix.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model/'))
    graph = tf.get_default_graph()
    infer_op = graph.get_operation_by_name('train_op')
    z = graph.get_tensor_by_name('z1:0')
    generated = graph.get_tensor_by_name('Sigmoid:0')
    img = sess.run(generated,feed_dict={z:z_batch})
    img = img*255
    img = img.astype(np.uint8)
    img = Image.fromarray(img[-1,:,:,:], 'RGB')
    img.show()
    

