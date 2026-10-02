import tensorflow as tf

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.1),

    tf.keras.layers.RandomZoom(0.15),

    tf.keras.layers.RandomContrast(0.2),

])

print("Data Augmentation Pipeline Created Successfully!")