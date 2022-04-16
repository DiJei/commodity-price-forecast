import torch
import torch.nn as nn

## LSTM CLASS
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, lookforward, bidirectional = False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lookforward = lookforward
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, bidirectional = bidirectional)
        self.fc = nn.Linear(hidden_dim, self.lookforward * self.output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        out = torch.reshape(out,(out.shape[0],self.lookforward,self.output_dim))
        return out
    
    
    
class BILSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, lookforward, bidirectional = True):
        super(BILSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lookforward = lookforward
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, bidirectional = bidirectional)
        self.fc = nn.Linear(hidden_dim*2, self.lookforward * self.output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        out = torch.reshape(out,(out.shape[0],self.lookforward,self.output_dim))
        return out