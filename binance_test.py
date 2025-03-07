import tensorflow as tf

# Load the existing model without compiling
model = tf.keras.models.load_model("lstm_model.h5", compile=False)

# Recompile the model with explicit loss function
model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])

# Save the model properly
model.save("lstm_model_fixed.h5")

print("✅ Model has been fixed and saved as 'lstm_model_fixed.h5'")
