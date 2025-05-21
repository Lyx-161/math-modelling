import numpy as np
from scipy.interpolate import Rbf
import torch
import torch.nn as nn
import torch.optim as optim

class BaseFitter:
    def __init__(self, t, x, y):
        self.t = t
        self.x = x
        self.y = y

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the fit method")

class PolynomialFitter(BaseFitter):
    def fit(self, degree,num_points=1000):
        x_coefficients = np.polyfit(self.t, self.x, degree)
        y_coefficients = np.polyfit(self.t, self.y, degree)
        
        # 生成拟合曲线
        t_fit = np.linspace(0, 1, num_points)
        x_fit = np.polyval(x_coefficients, t_fit)
        y_fit = np.polyval(y_coefficients, t_fit)

        return x_fit, y_fit

class SplineFitter(BaseFitter):
    def fit(self, s=1.0,num_points=1000):
        from scipy.interpolate import splev, splrep
        
        tck_x = splrep(self.t, self.x, s=s)
        tck_y = splrep(self.t, self.y, s=s)
        
        t_fit = np.linspace(0, 1, num_points)
        x_fit = splev(t_fit, tck_x)
        y_fit = splev(t_fit, tck_y)
        
        return x_fit, y_fit
    
class RBFFitter(BaseFitter):
    def fit(self, num_points=1000):
        rbf_x = Rbf(self.t, self.x, function='gaussian')
        rbf_y = Rbf(self.t, self.y, function='gaussian')
        
        # 生成拟合曲线的参数t
        t_fit = np.linspace(0, 1, num_points)
        
        # 计算拟合曲线
        x_fit = rbf_x(t_fit)
        y_fit = rbf_y(t_fit)
        
        return x_fit, y_fit
    
class NeuralNetworkFitter(BaseFitter):
    def __init__(self, t, x, y,criterion,function):
        super().__init__(t, x, y)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if function=="Relu":
            self.f=nn.ReLU()
        elif function=="Gelu":
            self.f=nn.GELU()
        elif function=="Elu":
            self.f=nn.ELU()
        self.f=nn.ELU()
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            self.f,
            nn.Linear(128, 64),
            self.f,
            nn.Linear(64, 32),
            self.f,
            nn.Linear(32, 2)
        ).to(self.device)
        
        if criterion=='mse':
            self.criterion=nn.MSELoss()
        elif criterion=='chamfer':
            self.criterion=self.chamfer_loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.01)

    def chamfer_loss(self,point_set1, point_set2):
        diff1 = point_set1.unsqueeze(1) - point_set2.unsqueeze(0)
        #print(diff1.shape)
        dist1 = torch.sum(diff1 ** 2, dim=2)

        min_dist1, _ = torch.min(dist1, dim=1)

        min_dist2, _ = torch.min(dist1, dim=0)
        chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
        return chamfer_dist


    def fit(self, num_points=1000, epochs=5000):
        # 准备数据
        t = torch.tensor(self.t, dtype=torch.float32).reshape(-1, 1).to(self.device)
        x = torch.tensor(self.x, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y, dtype=torch.float32).to(self.device)
        data = torch.cat((x.unsqueeze(1), y.unsqueeze(1)), dim=1)

        # 训练模型
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(t)
            loss = self.criterion(outputs, data)
            loss.backward()
            self.optimizer.step()

        # 生成拟合曲线的参数t
        t_fit = torch.linspace(0, 1, num_points).reshape(-1, 1).float().to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(t_fit)
        x_fit = predictions[:, 0].cpu().numpy()
        y_fit = predictions[:, 1].cpu().numpy()
        
        return x_fit, y_fit