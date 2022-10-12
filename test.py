import paddle
import paddle.incubate.sparse as sparse
import numpy as np 

x_indices = np.load("x_indices.npy")
x_values = np.load("x_values.npy")
shape = [4, 41, 1440, 1440, 5]
x = sparse.sparse_coo_tensor(x_indices, x_values, shape=shape)
weight = np.load("weight.npy")
weight = paddle.to_tensor(weight)
out = sparse.nn.functional.conv3d(x, weight, bias=None, stride=[2,2,2], padding=[0,0,0], dilation=[1,1,1], groups=1, cutlass=False)
np.save("out2_indices", out.indices().numpy())
np.save("out2_values", out.values().numpy())
print(out)
