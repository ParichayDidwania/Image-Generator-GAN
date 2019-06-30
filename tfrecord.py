import tensorflow as tf
import cv2


img_path1 = './images/b1.jpg'
img_path2 = './images/l1.jpg'
tf_path = './mix.tfrecords'

l=[]
l.append(img_path1)
l.append(img_path1)


l.append(img_path2)
l.append(img_path2)



def _bytes_feature(value):
    return tf.train.Feature(bytes_list =  tf.train.BytesList(value = [value]))

writer = tf.python_io.TFRecordWriter(tf_path)

for j in range(100):
    for i in l:
        print(i)
        with tf.gfile.FastGFile(i,'rb') as img:
            img1 = img.read()

        feature = {
            'image_1' : _bytes_feature(img1)
        }    
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example.SerializeToString())


