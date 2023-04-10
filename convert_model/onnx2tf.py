import onnx
from onnx_tf.backend import prepare

root = './weight'
file_name = 'new_best'
# Load the ONNX model
model = onnx.load(f'{root}/{file_name}.onnx')
tf_rep = prepare(model)
tf_rep.export_graph(f'{root}/{file_name}.pb')
print('onnx to pb success')
