import torch
import torch.nn as nn
import torch.optim as optim

class MultiTaskLoss(nn.Module):
    def __init__(self, eta):
        super(MultiTaskLoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.eta = nn.Parameter(torch.Tensor(eta))

    def forward(self, input, targets):
        loss = [self.loss_fn(o,y).sum() for o, y in zip(input, targets)]
        total_loss = torch.Tensor(loss) * torch.exp(-self.eta) + self.eta
        return loss, total_loss.sum() # omit 1/2

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.f1 = nn.Linear(5, 1, bias=False)
        self.f2 = nn.Linear(5, 1, bias=False)

    def forward(self, input):
        outputs = [self.f1(x).squeeze(), self.f2(x).squeeze()]
        return outputs

model = MultiTaskModel()
loss = MultiTaskLoss(eta=[2.0,1.0])

# mtl = MultiTaskLoss(model=MultiTaskModel(),
#                     loss_fn=[nn.MSELoss(), nn.MSELoss()],
#                     eta=[2.0, 1.0])

print(list(model.parameters()))
print(list(loss.parameters()))

params = list(model.parameters()) + list(loss.parameters())

x = torch.randn(3, 5)
y1 = torch.randn(3)
y2 = torch.randn(3)

optimizer = optim.SGD(params, lr=0.1)
optimizer.zero_grad()
out = model(x)
loss, total_loss = loss(out, [y1, y2])
print(loss, total_loss)
total_loss.backward()
optimizer.step()