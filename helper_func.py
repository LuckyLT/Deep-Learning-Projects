def load_image(filename, label):

    print(filename, label)
    pdb.set_trace()
    result_file = tf.io.read_file(filename)
    result_image = tf.image.decode_image(result_file)
    result_image = tf.image.resize(result_file, (224, 224))



