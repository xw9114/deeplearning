import torch
import torch.nn as nn
x=torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
target=torch.tensor([[3.0],[5.0],[7.0],[9.0],[11.0]])

model=nn.Linear(1,1)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
loss_fn=nn.MSELoss()
for epoch in range(1000):
    prediction=model(x)
    loss=loss_fn(prediction,target)
    loss.backward()
    if (epoch + 1) % 100 == 0:
        print(
            f"第{epoch + 1}轮："
            f"w={model.weight.item():.4f}, "
            f"b={model.bias.item():.4f}, "
            f"loss={loss.item():.4f}, "
            f"梯度={model.weight.grad.item():.4f}"
        )
    optimizer.step()
    optimizer.zero_grad()
with torch.no_grad():
    new_x=torch.tensor([[1.0]])
    new_prediction=model(new_x)
    print("更新后的权重：", model.weight.item())
    print("更新后的偏置：", model.bias.item())
    print("更新后的预测：", new_prediction)
