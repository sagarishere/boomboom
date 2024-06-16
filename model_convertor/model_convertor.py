import tensorflow as tf
import pickle

# Load the .h5 model
model = tf.keras.models.load_model('../results/models/my_own_model.h5')

# Save the model to a .pkl file
with open('../results/models/my_own_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the .pkl model
with open('../results/models/my_own_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)