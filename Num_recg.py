import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    
    #神经网络主体，输入28*28,中间三层64个节点
    def __init__(self): 
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    #全连接线性计算，套上激活函数增加非线性环节
    def forward(self, x): 
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

#导入数据
def get_data_loader(is_train): 
    to_tensor = transforms.Compose([transforms.ToTensor()]) #张量

    #下载MNIST数据集：导入目录(空为当前目录)，训练集/数据集
    data_set = MNIST("", is_train, transform=to_tensor, download=True) 
    
    #一个批次15张图片，随机打乱返回数据加载器
    return DataLoader(data_set, batch_size=15, shuffle=True) 

#评估神经网络识别正确率
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data: #按批次取出数据
            outputs = net.forward(x.view(-1, 28*28)) #计算预测值
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]: #对预测最大可能值进行比较
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    
    #导入训练集和测试集并初始化
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    
    #打印初始网络正确率，--p->1/10
    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):  #每个轮次
        for (x, y) in train_data:
            net.zero_grad()              #初始化
            output = net.forward(x.view(-1, 28*28))         #正向传播
            loss = torch.nn.functional.nll_loss(output, y)  #计算损失差值
            loss.backward()              #反向误差传播
            optimizer.step()             #优化网络参数
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for (n, (x, _)) in enumerate(test_data):
        if n > 2:  #随机抽取图像显示预测结果
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()

