import tensorflow as tf
import numpy as np

X = np.arange(1, 10000).reshape(-1, 1)
y = np.arange(10001, 20000).reshape(-1, 1)

X_train = tf.constant(X)
y_train = tf.constant(y)

X_min = tf.reduce_min(X_train)
X_max = tf.reduce_max(X_train)
y_min = tf.reduce_min(y_train)
y_max = tf.reduce_max(y_train)

X_normalized = (X_train - X_min) / (X_max - X_min)
y_normalized = (y_train - y_min) / (y_max - y_min)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,)),
    tf.keras.layers.Dense(1, input_shape=(1,)),
])

model.compile(
    loss="mean_absolute_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["mae"]
)

model.fit(X_normalized, y_normalized, epochs=5)

predictions_X = np.array([[1000]])
predictions_X_normalized = (predictions_X - X_min.numpy()) / (X_max.numpy() - X_min.numpy())

predictions_normalized = model.predict(predictions_X_normalized)
predictions = predictions_normalized * (y_max.numpy() - y_min.numpy()) + y_min.numpy()
print(f"Predictions: {predictions}")

model.save("save2.h5")
