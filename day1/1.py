import torch

x = torch.tensor([2.0])
target = torch.tensor([10.0])
w = torch.tensor([1.0],requires_grad=True)

prediction=x*w
loss=(prediction-target)**2
print("当前权重：", w)
print("模型预测：", prediction)
print("当前损失：", loss)
loss.backward()
print("权重的梯度：", w.grad)
print(loss)

with torch.no_grad():
    w-=0.1*w.grad
new_prediction=w*x
new_loss=(new_prediction-target)**2
print("更新后的权重：", w)
print("更新后的预测：", new_prediction)
print("更新后的损失：", new_loss)