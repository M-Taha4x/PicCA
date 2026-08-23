import torch
x=torch.tensor(2.0) #x=2
w1=torch.tensor(0.5,requires_grad=True) #w1=0.5
w2=torch.tensor(1.0,requires_grad=True) #w2=1.0
y=torch.tensor(3.0) #y=3.0

z1=w1*x #z1=2*0.5=1
a1=z1 #activation fxn(identity activation)
z2=w2*a1  #z2=1*1=1
y_hat=z2 #output layer

#Loss Function
loss=(y-y_hat)**2 #loss=(3-1)**2=4
print(f"Prediction: {y_hat:.2f}")
print(f"Loss: {loss:.2f}") 
#BackPropagte
loss.backward() #computes gradient of the loss fxn
print("dL/dw1: ",w1.grad.item())
print("dL/dw2: ",w2.grad.item())

lr=0.1 #learning weight
#Update Weights
with torch.no_grad():
    w1-=lr*w1.grad
    w2-=lr*w2.grad
 #reset gradient   
w1.grad.zero_()
w2.grad.zero_()

print("Updated w1: ",w1.item())
print("Updated w2: ",w2.item())
print(f"Loss={loss.item():.4f}, w1={w1.item():.4f}, w2={w2.item():.4f}")