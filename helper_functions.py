def load_filenames(dir_path):
    """
            Returns the filenames and their corresponding classes.
    """

    filenames = {
        'image_file': [],
        'image_class': []
    }

    for car_brand in os.listdir(dir_path):
        folder_name = os.path.join(dir_path, car_brand)

        filenames_in_class = [os.path.join(folder_name, x) for x in os.listdir(folder_name)]
        filenames['image_file'].extend(filenames_in_class)

        filenames["image_class"].extend([car_brand] * len(filenames_in_class))

    return (pd.DataFrame(filenames))


def get_image_dimensions(image_filename):
    """
    Returns the dimensions of the image (height, width, channels) in pixels.

    There are better methods which don't involve reading the entire image
    and loading it in memory but this is simple enough.
    """
    return imread(image_filename).shape


def load_images(image_filename, image_label):
    # pdb.set_trace()
    # read the file and then decode it
    result_file = tf.io.read_file(image_filename)
    result_image = tf.image.decode_jpeg(result_file)  # decode_jpeg
    # result_image = tf.image.convert_image_dtype(result_image, tf.float32) #convert to float32

    # resize the image
    result_image = tf.image.resize(result_image, (224, 224))  # resize the data
    result_image /= 255.0
    # def preprocess_image(x):
    #     """
    #     This is a stripped-down version of Keras' own imagenet preprocessing function,
    #     as the original one is throwing an exception
    #     """
    #     pdb.set_trace()
    #     backend = tf.keras.backend
    #
    #     # 'RGB'->'BGR'
    #     x = x[..., ::-1]
    #     mean = [103.939, 116.779, 123.68]
    #     std = None
    #
    #     mean_tensor = backend.constant(-np.array(mean))
    #
    #     # Zero-center by mean pixel
    #     if backend.dtype(x) != backend.dtype(mean_tensor):
    #         x = backend.bias_add(
    #             x, backend.cast(mean_tensor, backend.dtype(x)))
    #     else:
    #         x = backend.bias_add(x, mean_tensor)
    #     if std is not None:
    #         x /= std
    #     return x

    # image = preprocess_image(result_image)

    return result_image, image_label


def initialize_tf_dataset(data, labels, should_batch=True, should_repeat=True):
    # pdb.set_trace()
    dataset = tf.data.Dataset.from_tensor_slices((data.image_file.values, labels))
    dataset = dataset.map(load_images)
    dataset = dataset.shuffle(buffer_size=len(data))

    if should_batch:
        dataset = dataset.batch(BATCH_SIZE)
    else:
        dataset = dataset.batch(len(data))

    if should_repeat:
        dataset = dataset.repeat()
    return dataset
