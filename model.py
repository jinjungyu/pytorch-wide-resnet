import os
import torch
from torch import nn

class WideBlock(nn.Module): # 얕은 구조에서 사용. Resnet 18, 34
    def __init__(self,in_ch,out_ch,stride=1):
        super(WideBlock,self).__init__()
        # stride를 통해 이미지 크기 조정
        # 한 층의 첫 번째 블록의 시작에서 다운 샘플 (첫 번째 층 제외)
        self.residual = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(num_features=in_ch),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1,padding=1))

        self.shortcut = nn.Sequential()
        # stride가 1이 아니면 합 연산이 불가하므로 매핑
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch,out_ch,kernel_size=1,
                                      stride=stride,padding=0)

    def forward(self,x):
        x_residual = self.residual(x)
        x_shortcut = self.shortcut(x)
        out = x_residual + x_shortcut
        return out

class WRN(nn.Module):
    def __init__(self,depth,k,num_classes=10):
        # Cifar-10 : num_classes = 10
        super(WRN,self).__init__()
        self.block = WideBlock
        self.num_classes = num_classes
        self.depth = depth
        self.k = k
        self.N = (self.depth - 4) // 6

        self.in_ch = 16
        # 32x32x3 -> 32x32x16
        self.conv1 = nn.Conv2d(3,self.in_ch,3,stride=1,padding=1)
        # 32x32x16 -> 32x32x16*k
        self.layer1 = self.make_layers(16*k,self.N,stride=1)
        # 32x32x16 -> 16x16x32*k
        self.layer2 = self.make_layers(32*k,self.N,stride=2)
        # 16x16x32*k -> 8x8x64*k
        self.layer3 = self.make_layers(64*k,self.N,stride=2)
        self.bn = nn.BatchNorm2d(64*k)
        self.relu = nn.ReLU()
        # 8x8x64*k -> 1x1x64*k
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 1x1x64*k -> (64*k,)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*k,self.num_classes)

    def make_layers(self,out_c,N,stride=1):
        strides = [stride] + [1] * (N-1)
        blocks = []
        for i in range(N):
            blocks.append(self.block(self.in_ch,out_c,strides[i]))
            self.inplanes = out_c
        return nn.Sequential(*blocks)

    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def WRN_40_8():
    return WRN(40,8)

def WRN_40_10():
    return WRN(40,10)

def WRN_28_8():
    return WRN(28,8)

def WRN_28_10():
    return WRN(28,10)

def WRN_22_8():
    return WRN(22,8)

def WRN_22_10():
    return WRN(22,10)

def WRN_16_8():
    return WRN(16,8)

def WRN_16_10():
    return WRN(16,10)


# 모델 저장 함수
def save(ckpt_dir,net,optim,epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},
               os.path.join(ckpt_dir,f"model_epoch{epoch}.pth"))

# 모델 로드 함수
def load(ckpt_dir,net,optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net,optim,epoch
    ckpt_list = os.listdir(ckpt_dir)
    if ckpt_list == []:
        epoch = 0
        return net,optim,epoch
    ckpt_list.sort(key=lambda f:int(''.join(filter(str.isdigit,f))))

    dict_model = torch.load(os.path.join(ckpt_dir,ckpt_list[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(''.join(filter(str.isdigit,ckpt_list[-1])))
    return net,optim,epoch
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth',default=28)
    parser.add_argument('-k',default=10)
    args = parser.parse_args()
    depth = args.depth
    k = args.k
    model_name = f'WRN_{depth}_{k}'
    net = locals()[model_name]()
    print(net,type(net))