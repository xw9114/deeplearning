import torch
x = torch.tensor([2.0])
target = torch.tensor([10.0])
w=torch.tensor([2.0],requires_grad=True)
opimizer=torch.optim.SGD([w],lr=0.1)
for epoch in range(10):
    opimizer.zero_grad()
    prediction=x*w
    loss=(prediction-target)**2
    loss.backward()
    print(
        f"第{epoch + 1}轮："
        f"w={w.item():.4f}, "
        f"预测={prediction.item():.4f}, "
        f"loss={loss.item():.4f}, "
        f"梯度={w.grad.item():.4f}"
    )
    opimizer.step()
