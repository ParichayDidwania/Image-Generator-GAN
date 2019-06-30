import tensorflow as tf
import numpy as np
from PIL import Image 

def parse(serialized):

    feature = {
        'image_1' : tf.FixedLenFeature([],tf.string)
    }

    example = tf.parse_single_example(serialized = serialized, features = feature)
    img1 = example['image_1']
    img1 = tf.image.decode_image(img1)
    img1 = tf.reshape(img1,[28,28,3])
    img1 = tf.image.convert_image_dtype(img1,tf.float32)
    
    #img1 = tf.reshape(img1,[200,200,3])

    #img2 = np.random.uniform(-1,1,[200,200,3])

    return img1

def read_data(filename,batch_s = 1):

    dataset = tf.data.TFRecordDataset(filenames = filename)
    dataset = dataset.map(parse,16)
    dataset = dataset.prefetch(10)
    dataset = dataset.batch(batch_size = batch_s)

    return dataset
