import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true')
opt = parser.parse_args()
print(opt)
root = './weight'
file_name = 'new_best'
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(f'{root}/{file_name}.pb') # path to the SavedModel directory

if opt.fp16:
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  file_name += '_fp16' 
tflite_model = converter.convert()

# Save the model.
with open(f'{root}/{file_name}.tflite', 'wb') as f:
  f.write(tflite_model)