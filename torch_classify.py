'''
2023.3.31 danyow
使用pytroch框架实现车牌识别
ref:
1. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
2. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html ✅
3. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
4. https://www.kaggle.com/code/bitthal/understanding-input-data-and-loading-with-pytorch ✅
实现步骤：
1. 构建数据集类，使用xml的文件信息去提取车牌特征，转换尺寸之后输入神经网络

'''
import os 
import matplotlib.pyplot as plt 
from PIL import Image
import random
import xml.etree.ElementTree as ET
import torch
import torch.optim as optim 
from torchvision import transforms, datasets,utils
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import glob 
import torchvision 
import numpy as np 
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的

# 定义训练集和测试集的路径
trainData_dir = 'Train/images'
testData_dir = 'Test/images'
images= os.listdir(trainData_dir)
# print(f'文件夹下总共有{len(images)}条数据')

# 构建图片名称和索引的映射
name_map = {}
for i in range(len(images)):
    # print(images[i][:-4])
    name_map[i] =  images[i][:-4]

# print('名称索引映射',name_map)
print(len(name_map))

# 展示图片
# image = images[random.randint(0,len(images))]
# image = images[0]
# image_path= os.path.join(trainData_dir,image)
# print("文件路径",image_path)


# 引入Annotations文件来增强特征提取
trainAnno_dir = 'Train/annotations'
testAnno_dir = 'Train/annotations'

# # xml文件和图片文件名字相同所以使用上述image的名称【但是要把png转换为xml】
# annotation = image.replace('png','xml') # 字符串转换
# annotation_path = os.path.join(trainAnno_dir,annotation)
# print('注解路径',annotation_path)



def ImageLoader(path,trainAnno_dir,trainData_dir):
    '''输入文件名，自动去对应路径下的文件'''

    img_path = os.path.join(trainData_dir,path) 
    ann_path = os.path.join(trainAnno_dir,path).replace('png','xml')

    print(f'图片文件路径={img_path}，注解文件路径={ann_path}')

    img = Image.open(img_path)
    # 加载图片，从xml文件中获得bbox的数组然后返回图片矩阵
    tree = ET.parse(ann_path)
    root = tree.getroot()
    objects = root.findall('object')

    # 从xml中查看属性名称
    for obj in objects:
        bounding_box = obj.find('bndbox')
        # 方法2
        xmin = int(bounding_box[0].text) 
        ymin = int(bounding_box[1].text)
        xmax = int(bounding_box[2].text)
        ymax = int(bounding_box[3].text)  
        
    bbox = (xmin, ymin, xmax, ymax)
    # 转换尺寸
    img = img.crop(bbox)
    img = img.resize((64,64),Image.ANTIALIAS)
    img = np.array(img)

    return img 


# 测试ImageLoader
# img = ImageLoader('Cars0.png',trainAnno_dir,trainData_dir)
# print(type(img))


'''https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
class XMLDataset(Dataset):
    def __init__(self,trainData_dir,trainAnno_dir,name_map,transform):
        # 图像总路径
        self.img_path = trainData_dir
        # 注解总路径
        self.ann_path = trainAnno_dir
        # 总名称路径
        self.name_map = name_map
        self.transform = transform


        
    def __len__(self):
        '''返回数据集的总长度'''
        return len(self.name_map)

    def __getitem__(self, idx):
        '''以支持索引，使得数据集[i]可以用于获得第i个样本。'''
        # 输入索引，转换为名称
        idx = self.name_map[idx]
        img_name = idx+'.png'
        ann_name = idx+'.xml'

        # 图片完整路径
        img_path = os.path.join(self.img_path,img_name)
        # 注解完整路径
        ann_path = os.path.join(self.ann_path,ann_name)

        # 对图片进行xml特征提取
        img = Image.open(img_path).convert('RGB')


        # 加载图片，从xml文件中获得bbox的数组然后返回图片矩阵
        tree = ET.parse(ann_path)
        root = tree.getroot()
        objects = root.findall('object')
        # 从xml中查看属性名称
        for obj in objects:
            bounding_box = obj.find('bndbox')
            # 方法2
            xmin = int(bounding_box[0].text) 
            ymin = int(bounding_box[1].text)
            xmax = int(bounding_box[2].text)
            ymax = int(bounding_box[3].text)  
            # label = obj.find('name').text 
            # label = obj.find('truncated').text

        bbox = (xmin, ymin, xmax, ymax)
        # 转换尺寸
        img = img.crop(bbox)
        # print(img.size)
        img = img.resize((64,64))

        img = np.array(img,dtype=np.float32) # unsinged char 和float、
        '''
        swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        '''
        img = img.transpose((2, 0, 1))# 转换格式 
        # label = torch.tensor(label)

        # print(img.shape)# 64,64
        
        # # 都是同一个label
        # sample = {'image':img,'label':np.array([0])}
        if self.transform:
            sample = self.transform(img)
        
        # return sample
        return img,torch.tensor(0)
    
    
    
'''
需要重写torch的dataloader,
返回
    List of all classes and dictionary mapping each class to an index.
返回格式
    (Tuple[List[str], Dict[str, int]])
'''
# 定义类别字典
class_dict = {'licence':0}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': label}
    


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # print(image.shape)
        print("ToTensor--")
        image = image.transpose((2, 0, 1))
        image = image.astype('float32')
        label = label.astype('float32')
        
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
    
license_dataset = XMLDataset(
    trainData_dir,
    trainAnno_dir,
    name_map,
    transform = transforms.Compose([
        # Rescale(64),
        # ToTensor(),
        transforms.ToTensor(),
    ])
)
print('这一步ok》')
# for i in range(len(license_dataset)):
#     sample = license_dataset[i]

#     print(i, sample['image'].shape, sample['label'].shape)
 
#     if i == 5:
#         break


def show_crop_batch(sample_batched):
    '''使用裁剪后的图片进行展示'''
    images_batch,label_batch = sample_batched['image'],sample_batched['label']
    
    grid = utils.make_grid(images_batch)
    plt.title('返回批次中的裁剪后的图片信息')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

'''
然而，由于使用简单的for循环来迭代数据，我们正在失去许多功能。特别是，我们错过了：
- Batching the data
- Shuffling the data
- Load the data in parallel using multiprocessing workers.
'''
# 使用torch.utils.data.DataLoader来作为迭代器，
dataloader  = DataLoader(license_dataset,batch_size=4,shuffle=True,num_workers=0)

# 测试在批量中的图片信息
# for i_batch,sample_batched in enumerate(dataloader):
#     # print(i_batch, sample_batched['image'].size(),
#     #       sample_batched['label'].size())
    
#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
        
#         show_crop_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break



''' 定义模型'''
import torch.nn as nn 
import torch.nn.functional as F 


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        in_channels是输入的四维张量[N, C, H, W]中的C了，即输入张量的channels数。这个形参是确定权重等可学习参数的shape所必需的。
        
        torch的张量输入是[Num,channels,Height,Weight]

        '''

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)


        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):


        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
# 训练网络
N_EPOCHS=5 
for epoch in range(N_EPOCHS):
    running_loss = 0.0 
    # for i,data in enumerate([image,'licence'],0):
    for i,data in enumerate(dataloader,0):
        # print(type(data))# 我的返回是字典


        # 获得输入，data的形状是[inputs,labels]
        inputs,labels = data[0],data[1]
        
        # print(f'inputs_shape',inputs.shape)
        # print(f"inputs={inputs},labels={labels}")
        # print('inputs_shape,length of inputs',inputs.shape,len(inputs))
        # print(type(inputs),inputs.dtype) # 后者查看具体类型
        # inputs = inputs.astype('float')
        
        # 零梯度参数
        optimizer.zero_grad()

        # 前向传播+后向传播+优化
        outputs = net(inputs)
        # print(outputs)

        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        # 打印静态信息
        running_loss += loss.item()
        if i % 100 == 1:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')



