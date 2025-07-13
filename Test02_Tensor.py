import torch
import numpy as np

print("Torch版本：", torch.__version__)

scalar_a = torch.tensor(9)
scalar_b = torch.tensor(2)
vector = torch.tensor([1,2,5])
matrix = torch.tensor([[1,2],[3,4]])

print(f'标量：{scalar_a} 和 {scalar_b} {scalar_a.dtype} \n'
      f'向量：{vector} {vector.dtype} \n'
      f'矩阵：{matrix} {matrix.dtype} shape: {matrix.shape}\n'
      f'标量a+b= {scalar_a + scalar_b}'
      )
print('标量a*b=', scalar_a * scalar_b)
print('add:', torch.add(scalar_a,vector))
print('mult:', torch.mul(scalar_a,scalar_b))

x_ones = torch.ones_like(matrix)
print(x_ones)
x_rand = torch.rand_like(matrix, dtype=torch.float64)
print(x_rand)

shape = (2,3)
zeros_tensor = torch.zeros(shape)
print(f'{zeros_tensor}')

tensor1 = torch.rand(3,4)
print(f'{tensor1},tensor1 储存在：{tensor1.device}')

