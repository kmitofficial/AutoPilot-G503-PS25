# DriveLM/model.py

import torch
import torch.nn as nn

class IntentVLM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=10):
        super(IntentVLM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    @classmethod
    def from_pretrained(cls, model_name, input_dim=512, hidden_dim=256, output_dim=10):
        print(f"Loading pretrained model: {model_name}")
        # Here you could load real pretrained weights if available
        return cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Optional helper
def load_model(path):
    model = IntentVLM(input_dim=512, hidden_dim=256, output_dim=10)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
# import torch
# import torch.nn as nn
# from transformers import BertTokenizer, BertModel, BertConfig

# # model.py
# import torch
# import torch.nn as nn

# class IntentVLM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(IntentVLM, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x


