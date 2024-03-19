# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)), # 重置图像分辨率： 尺寸重新缩放为512，同时保持纵横比
        transforms.RandomCrop(256), # 依据给定的size随机裁剪：随机裁剪为256×256像素。
        transforms.ToTensor() #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]；注意事项：归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset): # 继承data.Dataset
    def __init__(self, root, transform): # Initialize file path or list of file names.
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]# Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        img = Image.open(os.path.join(self.root, path)).convert('RGB')#读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
        img = self.transform(img) #Preprocess the data (e.g. torchvision.Transform).
        return img

    def __len__(self):
        return len(self.paths) #返回数据长度

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count): #动态调整学习率
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)#自定义args.lr_decay衰减，iteration_count轮数、迭代数=epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser() #创建一个解析对象；然后向该对象中添加你要关注的命令行参数和选项
# Basic options （每一个add_argument方法对应一个你要关注的参数或选项）
parser.add_argument('--content_dir', type=str, default='../../数据集/train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='../../数据集/train',
                    help='Directory path to a batch of style images')

parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', 
                    help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./exp',      # 在exp文件夹中保存模型文件
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--stage1_iter', type=int, default=80000)
parser.add_argument('--stage2_iter', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=2)  # batch_size
parser.add_argument('--style_weight', type=float, default=1.0) #风格损失函数权重
parser.add_argument('--content_weight', type=float, default=1.0)#内容损失函数权重
parser.add_argument('--gan_weight', type=float, default=5.0) # 对抗损失函数权重
parser.add_argument('--identity_weight', type=float, default=50.0) #identity损失函数权重
# parser.add_argument('--AR1_weight', type=float, default=0.5) #损失函数权重
parser.add_argument('--AR2_weight', type=float, default=500.0) #损失函数权重
parser.add_argument('--n_threads', type=int, default=0) #多线程
parser.add_argument('--save_model_interval', type=int, default=10000)#interval间隔数，保存模型的间隔
parser.add_argument('--resume', action='store_true', help='enable it to train the model from checkpoints')  # 使其能够从检查点训练模型
parser.add_argument('--checkpoints', default='./checkpoints',
                    help='Directory to save the training checkpoints')
args = parser.parse_args() # 最后调用parse_args()方法进行解析，解析成功以后可以使用

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #使用gpu训练

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)#mkdir 创建文件夹，parents：如果父目录不存在，是否创建父目录。 # exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
checkpoints_dir = Path(args.checkpoints)
checkpoints_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder  # 解码器
vgg = net.vgg  # 编码器

discriminator = net.AesDiscriminator()  # 鉴别器



#  加载已经训练好的编码器模型参数
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, discriminator)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

# 定义 decoder/ transform优化器
optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()}], lr=args.lr)
# 定义 discriminator优化器
optimizer_D = torch.optim.Adam(network.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

start_iter = -1

# Enable it to train the model from checkpoints
if args.resume:
    checkpoints = torch.load(args.checkpoints + '/checkpoints.pth.tar')
    network.load_state_dict(checkpoints['net'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    start_iter = checkpoints['epoch']

# Training
for i in tqdm(range(start_iter+1, args.stage1_iter+args.stage2_iter)):  # （start_iter+1=0； args.stage1_iter+args.stage2_iter=80000+80000）
    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_D, iteration_count=i) 
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    
    # if i < args.stage1_iter: # i<80000
        # network的返回值是 g_t, loss_c, loss_s, loss_gan_d, loss_gan_g, l_identity, loss_aesthetic
    stylized_results, loss_c, loss_s, loss_gan_d, loss_gan_g, loss_id, loss_AR2 = network(content_images, style_images, aesthetic=True)
    # else:
    # stylized_results, loss_c, loss_s, loss_gan_d, loss_gan_g, loss_AR1, loss_AR2 = network(content_images, style_images)

    # train discriminator
    optimizer_D.zero_grad()  # 鉴别器优化器梯度清零
    loss_gan_d.backward(retain_graph=True)   # 反向传播，得到每个节点梯度


    # train generator
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s

    loss_gan_g = args.gan_weight * loss_gan_g

    # if i < args.stage1_iter:  # i<80000
    loss_id = args.identity_weight * loss_id
    #     loss = loss_c + loss_s + loss_gan_g + loss_id
    # else:
        # loss_AR1 = args.AR1_weight * loss_AR1
    loss_AR2 = args.AR2_weight * loss_AR2
    loss = loss_c + loss_s + loss_gan_g + loss_id + loss_AR2

    optimizer.zero_grad()  # 梯度清零
    loss.backward(retain_graph=True)  # 反向传播，得到每个节点梯度
    optimizer.step()  # 参数优化
    optimizer_D.step()  # 参数优化
    
    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_gan_g', loss_gan_g.item(), i + 1)
    writer.add_scalar('loss_gan_d', loss_gan_d.item(), i + 1)

    # Save intermediate results
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    # if (i + 1) % 1 == 0:
    if (i + 1) % 10000 == 0:
        visualized_imgs = torch.cat([content_images, style_images, stylized_results])
      
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        save_image(visualized_imgs, str(output_name), nrow=args.batch_size)
        print('[%d/%d] loss_content:%.4f, loss_style:%.4f, loss_gan_g:%.4f, loss_gan_d:%.4f' % (i+1, args.stage1_iter+args.stage2_iter, loss_c.item(), loss_s.item(), loss_gan_g.item(), loss_gan_d.item()))

    # Save models
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.stage1_iter+args.stage2_iter:
        checkpoints = {
            "net": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": i
        }
        torch.save(checkpoints, checkpoints_dir / 'checkpoints.pth.tar')
        
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth'.format(i + 1))

        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'transformer_iter_{:d}.pth'.format(i + 1))

        state_dict = network.discriminator.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'discriminator_iter_{:d}.pth'.format(i + 1))

writer.close()
