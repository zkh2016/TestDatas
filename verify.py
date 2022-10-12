import numpy as np

#prefix = './dump_data/'
prefix = ''

out = np.load(prefix + "conv_input_out_values.npy")
out_cutlass = np.load(prefix + "cutlass_conv_input_out_values.npy")
print(np.allclose(out, out_cutlass, rtol=1e-1))

out = np.load(prefix + "x_conv1_out_values.npy")
out_cutlass = np.load(prefix + "cutlass_x_conv1_out_values.npy")
print (np.allclose(out, out_cutlass, rtol=1e-1))

out = np.load(prefix + "x_conv2_out_values.npy")
out_cutlass = np.load(prefix + "cutlass_x_conv2_out_values.npy")
print (np.allclose(out, out_cutlass, rtol=1e-1))

out = np.load(prefix + "x_conv3_out_values.npy")
out_cutlass = np.load(prefix + "cutlass_x_conv3_out_values.npy")
print (np.allclose(out, out_cutlass, rtol=1e-5))

out = np.load(prefix + "x_conv4_out_values.npy")
out_cutlass = np.load(prefix + "cutlass_x_conv4_out_values.npy")
print (np.allclose(out, out_cutlass, rtol=1e-5))

out = np.load(prefix + "out_values.npy")
out_cutlass = np.load(prefix + "cutlass_out_values.npy")
print (np.allclose(out, out_cutlass, rtol=1e-5))


