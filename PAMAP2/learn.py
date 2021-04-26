# -*- coding: utf-8 -*-
import torch
import math

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
m = torch.linspace(math.pi, 2 * math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
# a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
# b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
# c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
# d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

a = torch.tensor(0.0107, requires_grad=True)
b = torch.tensor(-0.4134, requires_grad=True)
c = torch.tensor(0.6150, requires_grad=True)
d = torch.tensor(0.0953, requires_grad=True)
print(a, b, c, d)
learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    # loss = (y_pred - y).pow(2).sum()
    y_pred.backward()
    print(a.grad, b.grad, c.grad, d.grad)
    # # tensor(8143.9741) tensor(-2002.2303) tensor(48162.1562) tensor(4761.4971)
    y_pred_m = a + b * m + c * m ** 2 + d * m ** 3
    # loss_m = (y_pred_m - y).pow(2).sum()
    y_pred_m.backward()
    print(a.grad, b.grad, c.grad, d.grad)
    # tensor(93231.8516) tensor(476445.3125) tensor(2501040.5000) tensor(13429421.)
    if t % 100 == 99:
        print(t, y_pred.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.


    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
