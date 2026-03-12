import torch
import torch.nn as nn
class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(VolatilityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Lớp LSTM: Học các phụ thuộc chuỗi thời gian
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Lớp Dropout: Chống overfitting (rất quan trọng vì data tài chính cực nhiễu)
        self.dropout = nn.Dropout(0.2)
        # Lớp Fully Connected: Chuyển đổi hidden state thành giá trị dự báo cụ thể
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # Khởi tạo hidden state và cell state ban đầu
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward qua LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Chúng ta chỉ lấy output của bước thời gian cuối cùng (last time step)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
# Khởi tạo model thử nghiệm
model = VolatilityLSTM(input_size=5, hidden_size=64, num_layers=2)
print(model)