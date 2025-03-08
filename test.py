import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load preprocessed data directly
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Load the trained model
model_path = "optimized_5m_lstm_model.h5"
custom_objects = {"mse": MeanSquaredError()}
model = load_model(model_path, custom_objects=custom_objects)

# Ensure input shape matches the model
print(f"Model expects input shape: {model.input_shape}")
print(f"X_test shape: {X_test.shape}")

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate model performance
loss, mae = model.evaluate(X_test, y_test, verbose=0)

# Display evaluation results
print(f"🔹 Model Evaluation Results:")
print(f"   - Loss (MSE): {loss:.4f}")
print(f"   - Mean Absolute Error (MAE): {mae:.4f}")

# Convert predictions and actual values to a DataFrame
results_df = pd.DataFrame({"Actual Price": y_test.flatten(), "Predicted Price": y_pred.flatten()})

# Save results for further analysis
results_df.to_csv("model_predictions.csv", index=False)

print("\n✅ Model predictions saved to 'model_predictions.csv'. Open it to analyze results.")
