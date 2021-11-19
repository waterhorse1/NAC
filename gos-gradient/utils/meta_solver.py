import torch
import torch.nn as nn
import torch.nn.functional as f

class conv1d_small(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim,8,5,1,2, bias=False)
        self.conv2 = nn.Conv1d(8,16,5,1,2, bias=False)
        self.conv3 = nn.Conv1d(16,8,5,1,2, bias=False)
        self.conv4 = nn.Conv1d(8,1,5,1,2, bias=False)
    def forward(self, x):
        # x i * 1 * n
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        
        return x

class meta_solver_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_local = conv1d_small(2)
        self.conv1d_global = conv1d_small()
    def forward(self, x):
        #x 1 * n * n
        #print(x.shape)
        x = x.squeeze(dim=0)
        n = x.shape[-1]
        x = torch.transpose(x, 0, 1)#n * 1 * n
        global_feature = torch.mean(self.conv1d_global(x), dim=0)# 1 * 1 * n
        x = torch.cat([x,global_feature.repeat(n,1,1)], dim=1)#torch.cat([x, global_feature.repeat([n,1,1])], dim=-1)#n * 1 * n
        x = self.conv1d_local(x)
        output = torch.transpose(x, 0, 1)
        #print(output.shape)
        pi_1 = f.softmax(output.mean(dim=-1), dim=-1)
        return pi_1

class conv1d_large(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim,16,5,1,2, bias=False)
        self.conv2 = nn.Conv1d(16,32,5,1,2, bias=False)
        self.conv3 = nn.Conv1d(32,16,5,1,2, bias=False)
        self.conv4 = nn.Conv1d(16,1,5,1,2, bias=False)
    def forward(self, x):
        # x i * 1 * n
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        
        return x

class meta_solver_large(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_local = conv1d_large(2)
        self.conv1d_global = conv1d_large()
    def forward(self, x):
        #x 1 * n * n
        #print(x.shape)
        x = x.squeeze(dim=0)
        n = x.shape[-1]
        x = torch.transpose(x, 0, 1)#n * 1 * n
        global_feature = torch.mean(self.conv1d_global(x), dim=0)# 1 * 1 * n
        x = torch.cat([x,global_feature.repeat(n,1,1)], dim=1)#torch.cat([x, global_feature.repeat([n,1,1])], dim=-1)#n * 1 * n
        x = self.conv1d_local(x)
        output = torch.transpose(x, 0, 1)
        #print(output.shape)
        pi_1 = f.softmax(output.mean(dim=-1), dim=-1)
        return pi_1

class meta_solver_gru(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,32)
        self.gru1 = nn.GRU(32, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        
        self.final = nn.Sequential(nn.Linear(32*2,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,1))
        
    def forward(self, x):
        x = x[0]
        n = x.shape[-1]
        x = x.unsqueeze(dim=-1)
        x1 = self.l1(x)[0] #1*n*n*1->1*n*n*64->1*n*256
        o, h = self.gru1(x1) #1*n*128
        x_local = h #1*n*64
        o, h = self.gru2(h)# h 1*1*64
        x_global = h.repeat(1,n,1)
        x_final = torch.cat([x_local, x_global], dim=-1)#1*n*128
        output = f.softmax(self.final(x_final).squeeze(dim=-1)/1.0, dim=-1)
        return output
    
class meta_solver_gru_temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,32)
        self.gru1 = nn.GRU(32, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        
        self.final = nn.Sequential(nn.Linear(32*2,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,1))
        
    def forward(self, x):
        x = x[0]
        n = x.shape[-1]
        x = x.unsqueeze(dim=-1)
        x1 = self.l1(x)[0] #1*n*n*1->1*n*n*64->1*n*256
        o, h = self.gru1(x1) #1*n*128
        x_local = h #1*n*64
        o, h = self.gru2(h)# h 1*1*64
        x_global = h.repeat(1,n,1)
        x_final = torch.cat([x_local, x_global], dim=-1)#1*n*128
        output = f.softmax(self.final(x_final).squeeze(dim=-1)/1.0, dim=-1)
        return output