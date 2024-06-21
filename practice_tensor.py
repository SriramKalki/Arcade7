# (Originally a .ipynb for me to practice using tensors)

import torch

# Create two tensors
tensor_a = torch.tensor([1.0, 2.0, 3.0])
tensor_b = torch.tensor([4.0, 5.0, 6.0])

# Addition of tensors
tensor_sum = torch.add(tensor_a, tensor_b)
print(f"Sum of tensors:\n{tensor_sum}\n")

# Element-wise multiplication of tensors
tensor_product = torch.mul(tensor_a, tensor_b)
print(f"Element-wise product of tensors:\n{tensor_product}\n")

# Matrix multiplication
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[5, 6], [7, 8]])
matrix_product = torch.mm(matrix_a, matrix_b)
print(f"Matrix product:\n{matrix_product}\n")

# Compute the mean of a tensor
tensor_mean = torch.mean(tensor_a)
print(f"Mean of tensor_a: {tensor_mean}\n")

# Compute the sum of all elements in a tensor
tensor_sum_all = torch.sum(tensor_a)
print(f"Sum of all elements in tensor_a: {tensor_sum_all}\n")

# Applying an activation function (ReLU)
tensor_c = torch.tensor([-1.0, 0.0, 1.0])
relu_result = torch.nn.functional.relu(tensor_c)
print(f"ReLU applied to tensor_c:\n{relu_result}\n")

# Creating a tensor with requires_grad=True for autograd
tensor_d = torch.tensor([2.0, 3.0], requires_grad=True)
tensor_e = torch.tensor([4.0, 5.0], requires_grad=True)

# Define a simple computational graph
result = tensor_d * tensor_e
result_sum = torch.sum(result)

# Perform backpropagation
result_sum.backward()
print(f"Gradients for tensor_d: {tensor_d.grad}\n")
print(f"Gradients for tensor_e: {tensor_e.grad}\n")
