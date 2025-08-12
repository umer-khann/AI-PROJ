import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.activations import gelu # type: ignore
from tensorflow.keras import mixed_precision # type: ignore

# Load data
data = pd.read_csv("data/dataset.csv")

# Define input and output columns
input_cols = ['Angle', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 'TrackPosition'] + [f'Track_{i}' for i in range(1,19)]
output_cols = ['Acceleration', 'Braking', 'Clutch', 'Gear', 'Steering']

# Extract features and labels
X = data[input_cols].values
Y = data[output_cols].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
os.makedirs("preprocessing/processed", exist_ok=True)
with open("preprocessing/processed/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split dataset
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Mixed precision for efficiency 
mixed_precision.set_global_policy('mixed_float16')

# Build model with functional API
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(512, kernel_initializer='he_normal')(input_layer)
x = Activation(gelu)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, kernel_initializer='he_normal')(x)
x = Activation(gelu)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, kernel_initializer='he_normal')(x)
x = Activation(gelu)(x)

x = Dense(64, kernel_initializer='he_normal')(x)
x = Activation(gelu)(x)

x = Dense(32, kernel_initializer='he_normal')(x)
x = Activation(gelu)(x)

x = Dense(16, kernel_initializer='he_normal')(x)
x = Activation(gelu)(x)

x = Dense(8, kernel_initializer='he_normal')(x)
x = Activation(gelu)(x)

output_layer = Dense(len(output_cols), dtype='float32')(x)  # Force output to float32
model = Model(inputs=input_layer, outputs=output_layer)

# Custom weighted MSE loss
@tf.function
def weighted_mse(y_true, y_pred):
    weights = tf.constant([1.0, 1.0, 0.5, 2.0, 2.0], dtype=tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred) * weights, axis=-1)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=weighted_mse, metrics=['mae'])

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Setup callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ModelCheckpoint(
        filepath='model/torcs_nn_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=200,
          batch_size=64,
          callbacks=callbacks)

print("Training complete. Best model saved to model/torcs_nn_model.keras.")
