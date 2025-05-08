import torch
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)
# Load your model
model = SimpleMLP()
model.load_state_dict(torch.load("small_nn_acc73_f167.pt"))
model.eval()

# Script it (use trace if your model is purely tensor-based)
example = torch.randn(1, 3)
scripted_model = torch.jit.script(model, example)

# Save to file
scripted_model.save("fnn_temp_sun_rain.pt")
