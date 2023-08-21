# Disable all GPUs. This prevents errors caused by all workers trying to use the same GPU. In a real-world application, each worker would be on a different machine.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os
import logging
import multiprocessing
import tensorflow as tf
import tensorflow_datasets as tfds
# tf.get_logger().setLevel(logging.INFO)

# ---- who do what
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
print("cluster_resolver", cluster_resolver)
# -- set GPU for worker
def set_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus is None or len(gpus) == 0:
        print("No GPU!!\n")
        exit(0)
    else:
        # Restrict TensorFlow to only use the first GPU
        try:
            for device in gpus:
                tf.config.experimental.set_memory_growth(device, True)
            # tf.config.set_logical_device_configuration(
            #         gpus[0],
            #         [tf.config.LogicalDeviceConfiguration(memory_limit=3024)])
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print()
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            print()
            cpu_ph = tf.config.list_physical_devices('CPU')
            cpu_lg = tf.config.list_logical_devices('CPU')
            print(len(cpu_ph), "Physical CPUs,", len(cpu_lg), "Logical CPU")

        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

# if cluster_resolver.task_type in ("worker", "ps"):
set_gpu() # for all

# -- wait for task for worker and ps
if cluster_resolver.task_type in ("worker", "ps"):
    # Start a TensorFlow server and wait.
    # Set the environment variable to allow reporting worker and ps failure to the
    # coordinator. This is a workaround and won't be necessary in the future.
    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    # # Workers need some inter_ops threads to work properly.
    worker_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU':1})
    if cluster_resolver.task_type in ("worker"):
        NUM_WORKERS=len(cluster_resolver.cluster_spec().job_tasks('worker'))
        if multiprocessing.cpu_count() < NUM_WORKERS + 1:
            worker_config.inter_op_parallelism_threads = NUM_WORKERS + 1

    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        config=worker_config,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)
    print("cluster_resolver.task_type", cluster_resolver.task_type)
    print("cluster_resolver.task_id", cluster_resolver.task_id)
    print("cluster_resolver.rpc_layer", cluster_resolver.rpc_layer or "grpc")
    print("server.default_session_config", server.server_def.default_session_config)
    print()
    server.join()
elif cluster_resolver.task_type == "evaluator":   # Run sidecar evaluation
    pass # note used
else:  # Run the coordinator.
    # ---- ParameterServerStrategy object. will use all the available GPUs on each worker
    NUM_PS=len(cluster_resolver.cluster_spec().job_tasks('ps'))
    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=(256 << 10),
            max_shards=NUM_PS))

    strategy = tf.distribute.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner)

    # -- data
    # mnist = tfds.load('mnist', split='train', shuffle_files=False, data_dir="/workspace/mnist")
    # (x_train, y_train), (x_test, y_test) = mnist.load_data(data_dir="/workspace/mnist")
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    # -- trivial model
    with strategy.scope(): # dataset_fn will be wrapped into a tf.function and then executed on each worker to generate the data pipeline.
        # with tf.device('/device:GPU:0'):
        batch_size=32

        def normalize_img(data):
            """Normalizes images: `uint8` -> `float32`."""
            tf.print("Dataset.map(e_s_s)", tf.reduce_sum(data['image']))
            image = data['image']
            label = data['label']
            return tf.cast(image, tf.float32) / 255., label

        # -- Dataset TF class
        # suppress warning at worker, maybe fix error.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = tfds.load('mnist', split='train', shuffle_files=False, data_dir='/datasets/tensorflow_datasets')
        train_dataset = train_dataset.with_options(options)
        train_dataset = train_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(600).repeat().batch(batch_size).prefetch(300)

        # train_dataset = strategy.experimental_distribute_dataset(train_dataset)

        # -- model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(400, activation='relu'),
            # tf.keras.layers.Dense(3420, activation='relu'),
            # tf.keras.layers.Dense(3420, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # model.compile(tf.keras.optimizers.legacy.SGD(), RMSprop() loss="mse", steps_per_execution=10)
        model.compile(tf.keras.optimizers.legacy.Adam(), loss=loss_fn, steps_per_execution=10)
        # model.compile(optimizer='adam',
        #               loss=loss_fn,
        #               metrics=['accuracy'],
        #               # not required: pss_evaluation_shards='auto'
        #               )
        # print model
        model.summary()

    # -- train
    model.fit(train_dataset, epochs=3, steps_per_epoch=100)
    # -- save
    model.save('aa.keras', overwrite=True, save_format="tf")  # The file needs to end with the .keras extension save_format="tf"
    model = tf.keras.models.load_model('aa.keras', compile=False, safe_mode=False)
    validation_dataset = tfds.load('mnist', split='test', shuffle_files=False, data_dir='/datasets/tensorflow_datasets')
    validation_dataset = validation_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.shuffle(600).batch(batch_size)
    # -- checks the model's performance
    model.evaluate(validation_dataset, verbose=2)
    # -- inferece
    validation_dataset = validation_dataset.rebatch(1)
    # print(validation_dataset.__iter__().next())
    image, label = validation_dataset.__iter__().next()
    predictions = model(image).numpy()
    import numpy as np
    print(np.argmax(predictions))
    print(label)
