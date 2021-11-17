# SparseModule

## Summary
Strong Lottery Ticket Hypothesis ([Ramanujan et al. 2020](https://github.com/allenai/hidden-networks), [Malach et al. 2020](https://arxiv.org/abs/2002.00585), ...) states that randomly initialized neural networks already contain subnetworks with surprisingly good accuracy.
`SparseModule` enables us to find such subnetworks in any neural network architectures.

## How to Use

Wrap your network in `SparseModule`. That's it!

```python
net = nn.Linear(7,5,bias=False)

# All parameters in net are randomly initialized and fixed.
# model has score parameters, which is latent variables for subnetwork masks.
sparse_net = SparseModule(net, 0.8)
sparse_net = sparse_net.to(device)

# sparse_net.parameters() returns the score parameters.
# Never return the original parameters in net.
optimizer = optim.Adam(sparse_net.parameters(), lr=0.1) 

criterion = nn.MSELoss()
for i in range(10):
    optimizer.zero_grad()
    input = torch.randn(3,7).to(device)
    target = torch.randn(3,5).to(device)

    # Forward computation with masked net.
    output = sparse_net(input)
    loss = criterion(output, target)
    loss.backward()

    # Train score parameters (and thus masks).
    optimizer.step()
```

## Requirements
We've checked the code is valid under the following settings:
- Python 3.7.7
- PyTorch 1.5.0

