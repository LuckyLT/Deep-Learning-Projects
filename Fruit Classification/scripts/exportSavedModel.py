import tensorflow as tf

model_path = "Fruit Classification/saved_models/miniVGGNET"
export_path = "Fruit Classification/serving"

new_model = tf.keras.models.load_model(model_path)

# Check its architecture
new_model.summary()

#save model as saved model
tf.saved_model.save(new_model, export_path)