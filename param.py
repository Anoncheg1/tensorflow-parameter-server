import tensorflow as tf
import numpy as np
import os
import logging
import multiprocessing
# import sys
# import importlib

tf.get_logger().setLevel('INFO')

# -- model mapping "model name" for imclassif.py
select_model = {"resnet": "ResNet",
                "mobilenet": "MobileNet"}
SELECTED_MODEL="resnet"
# SELECTED_MODEL="mobilenet"
os.environ['MODEL_NAME'] = select_model[SELECTED_MODEL]

# -- who do what
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

# -- set GPU for worker
def set_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus is None or len(gpus) == 0:
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
            # print()
            # cpu_ph = tf.config.list_physical_devices('CPU')
            # cpu_lg = tf.config.list_logical_devices('CPU')
            # print(len(cpu_ph), "Physical CPUs,", len(cpu_lg), "Logical CPU")

        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            logging.exception(e)

# if cluster_resolver.task_type in ("worker", "ps"):
# set_gpu() # for all

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

# -- used for inter_ops threads and Dataset sharding.
NUM_WORKERS=len(cluster_resolver.cluster_spec().job_tasks('worker'))
# -- wait for task for worker and ps
if cluster_resolver.task_type in ("worker", "ps"):
    # Start a TensorFlow server and wait.

    # Workers need some inter_ops threads to work properly.
    worker_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU':1})
    if cluster_resolver.task_type in ("worker"):
        if multiprocessing.cpu_count() < NUM_WORKERS + 1:
            worker_config.inter_op_parallelism_threads = NUM_WORKERS + 1
    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        config=worker_config,
        protocol=cluster_resolver.rpc_layer or "grpc",
        start=True)
    tf.get_logger().info("cluster_resolver.task_type: " + str(cluster_resolver.task_type))
    tf.get_logger().info("cluster_resolver.task_id: " + str(cluster_resolver.task_id))
    tf.get_logger().info("cluster_resolver.rpc_layer: " + str(cluster_resolver.rpc_layer or "grpc"))
    tf.get_logger().info("server.default_session_config: " + str(server.server_def.default_session_config))
    server.join()
elif cluster_resolver.task_type == "evaluator":   # Run sidecar evaluation
    pass # note used
else:  # Run the coordinator.
    # ---- ParameterServerStrategy object. will use all the available GPUs on each worker
    NUM_PS=len(cluster_resolver.cluster_spec().job_tasks('ps'))
    tf.get_logger().info("NUM_PS: " + str(NUM_PS))
    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=(256 << 10),
            max_shards=NUM_PS))
    # tf.get_logger().info(f"FixedShardsPartitioner(num_shards={NUM_PS})")
    # variable_partitioner = tf.distribute.experimental.partitioners.FixedShardsPartitioner(num_shards=NUM_PS)
    strategy = tf.distribute.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner)
    # ---------------------------------------------------------------------------------------------------
    # ----------------------- Model, Dataset, Training --------------------------------------------------
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    tf.get_logger().info("options.experimental_distribute.auto_shard_policy: " + str(options.experimental_distribute.auto_shard_policy))

    with strategy.scope():
        # -- model loadling
        # m = importlib.import_module(select_model["mobilenet"])
        # m = importlib.import_module("imclassif")
        import imclassif as m # model, train_dataset, x_valid, y_valid, BATCH_SIZE, encode_single_sample, class_weights
        train_dataset = m.train_dataset
        train_dataset = train_dataset.shuffle(100) # cache and shuffle
        # train_dataset = train_dataset.with_options(options) # spread
        # train_dataset = train_dataset.shard(NUM_WORKERS, cluster_resolver.task_id) # separate cached to workers
        train_dataset = train_dataset.map(lambda x, y: m.encode_single_sample(x, y, [m.IMG_HEIGHT, m.IMG_WIDTH], False), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(m.BATCH_SIZE).prefetch(2)

    # ---- Train ----
    steps = m.y_train.shape[0]//m.BATCH_SIZE
    tf.get_logger().info(f"steps in epoch: {steps}, batch_size: {m.BATCH_SIZE}")
    m.model.fit(train_dataset, class_weight=m.class_weight, epochs=1, steps_per_epoch=steps)
    # m.model.fit(train_dataset, class_weight=m.class_weight, epochs=2)
    # -- Save model --
    m.model.save('savedmodel.keras', overwrite=True, save_format="tf")  # The file needs to end with the .keras extension
    # # -- Load model --
    # model = tf.keras.models.load_model('savedmodel.keras')
    # # -- Checks the model's performance --
    # print("evaluate")
    # validation_dataset = tf.data.Dataset.from_tensor_slices((m.x_valid.astype(str), m.y_valid.astype(int)))
    # validation_dataset = validation_dataset.map(lambda x, y: m.encode_single_sample(x, y), tf.data.experimental.AUTOTUNE)
    # validation_dataset = validation_dataset.batch(m.BATCH_SIZE).prefetch(100)

    # model.evaluate(validation_dataset, verbose=2)
    # # -- Inference --
    # print("inference", m.x_valid[0], m.y_valid[0])
    # im, l = m.encode_single_sample(m.x_valid[0], m.y_valid[0])
    # im = tf.expand_dims(im, axis=0)
    # print("image.shape", im.shape)
    # predictions = model.predict(im, batch_size=1)
    # print(np.argmax(predictions))
    # print("label:", m.y_valid[0])
    # exit(0)
