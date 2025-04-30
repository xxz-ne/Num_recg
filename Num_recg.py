import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from PIL import Image
import os

# 是否开启训练
TRAIN = True
MODEL_PATH = "mnist_model.pth"

class Net(torch.nn.Module):
    def __init__(self): 
        super().__init__()

        # 神经网络主体，输入28*28,中间三层64个节点
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

        # 添加Dropout神经元丢弃
        #self.dropout = torch.nn.Dropout(0.1)  
    
    #全连接线性计算，套上激活函数增加非线性环节
    def forward(self, x): 
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        #x = self.dropout(x)  # Dropout后再经过下一层
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)  
        return x

# 加载数据
def get_data_loader(is_train): 
    to_tensor = transforms.Compose([transforms.ToTensor()]) #张量

    # 下载MNIST数据集：导入目录(空为当前目录)，训练集/数据集
    data_set = MNIST("", is_train, transform=to_tensor, download=True)

    # 一个批次20张图片，随机打乱返回数据加载器
    return DataLoader(data_set, batch_size=20, shuffle=True)

# 评估神经网络识别正确率
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:   # 按批次取出数据
            outputs = net.forward(x.view(-1, 28*28)) # 计算预测值
            predictions = torch.argmax(outputs, dim=1)  # 对预测最大可能值进行比较
            n_correct += (predictions == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total

# 主训练流程
def train_model():

    # 导入训练集和测试集并初始化
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    # 打印初始网络正确率，--p->1/10
    print("initial accuracy:", evaluate(test_data, net))

    # 添加 L2 正则化，通过 weight_decay 参数
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4) #Adam优化
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) #SGD优化

    accuracy_list = []
    for epoch in range(10):  # 每个轮次
        for (x, y) in train_data:
            net.zero_grad()    # 初始化
            output = net.forward(x.view(-1, 28*28))   # 正向传播
            loss = torch.nn.functional.cross_entropy(output, y)   # 使用CrossEntropyLoss计算损失差值
            loss.backward()              # 反向误差传播
            optimizer.step()             # 优化网络参数
        acc = evaluate(test_data, net)
        accuracy_list.append(acc)
        print(f"epoch {epoch+1}, accuracy: {evaluate(test_data, net)}")

    # 绘制训练准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy over Epochs")
    plt.grid(True)
    plt.legend()
    plt.savefig("train_accuracy.png")
    plt.show()

    # 保存模型
    print("Training complete.")
    torch.save(net.state_dict(), "mnist_model.pth")
    print("Model saved to mnist_model.pth.")

    # 训练完毕随机抽取图像显示预测结果
    #for (n, (x, _)) in enumerate(test_data):
    #    if n > 2:
    #        break
    #    prediction = torch.argmax(net.forward(x[0].view(-1, 28*28)))
    #    plt.figure(n)
    #    plt.imshow(x[0].view(28, 28), cmap="gray")
    #    plt.title("prediction: " + str(int(prediction)))
    #plt.show()

# 图像预处理函数
def preprocess_image(image_path):
    
    # 打开图像并转换为灰度
    img = Image.open(image_path).convert('L')
    
    # 转换为28x28大小，符合MNIST标准
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 转换为Tensor，注意ToTensor会自动将图像值归一化到[0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),           # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 标准化
    ])
    img = transform(img)  # 这里img是一个[1, 28, 28]的Tensor
    return img.unsqueeze(0)  # 扩展维度成为[1, 1, 28, 28]，适配模型输入

# 使用模型对自定义图像进行预测
def predict_image(image_path, model):
 
    # 预处理图像
    img_tensor = preprocess_image(image_path)
    
    # 预测
    with torch.no_grad():
        output = model(img_tensor.view(-1, 28*28))  # 将图片张量展平
        prediction = torch.argmax(output, dim=1).item()
    
    return prediction

# 展示图像和预测结果
def display_image_and_prediction(image_path, prediction):
    
    img = Image.open(image_path).convert('L')
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {prediction}")
    plt.show()

# 示例：输入自定义图像进行识别
def recognize_digit(image_path):
    
    # 加载训练好的模型
    net = Net()
    # 可使用已保存的模型
    net.load_state_dict(torch.load("mnist_model.pth")) 
    net.eval()

    # 使用模型对自定义图像进行预测
    prediction = predict_image(image_path, net)
    display_image_and_prediction(image_path, prediction)
    print(f"Predicted digit: {prediction}")

if __name__ == "__main__":

    if TRAIN or not os.path.exists(MODEL_PATH):
        print("Training model...")
        train_model()
    else:
        print("Using existing trained model...")

    # 识别手写图像
    recognize_digit('./mywrite3.png')
    recognize_digit('./mywrite2.png')
    recognize_digit('./mywrite1.png')