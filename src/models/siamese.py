import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetGeneric(nn.Module):
    def __init__(self):
        super(SiameseNetGeneric, self).__init__()
        
    def _feed_fw(self, img) -> NotImplementedError:
        raise NotImplementedError("Must Implement Forward Propagation")
        
    def forward(self, img1, img2):
        feature1 = self._feed_fw(img1)
        feature2 = self._feed_fw(img2)
        return feature1, feature2
    
class ConvSiameseNet(SiameseNetGeneric):
    def __init__(self):
        super(ConvSiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(256, 2)
        
        self.pool = nn.MaxPool2d(kernel=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
    def _feed_fw(self, imgs):
        # x = 1, 28, 28
        x = self.dropout(self.pool(F.relu(self.conv1(imgs))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = x.reshape(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
if __name__ == "__main__":
    img1 = torch.randn((1, 28, 28))
    img2 = torch.randn((1, 28, 28))
    sample_model = ConvSiameseNet()
    f1, f2 = sample_model(img1, img2)
    print(f1.size())
    print(f2.size())