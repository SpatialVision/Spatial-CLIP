import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
resize = transforms.Resize((224, 224))

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'/home/maiyubo/llm/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('/home/maiyubo/llm/AlphaCLIP-main/train/dataset/data/depth/a3.jpg')
# import pdb;pdb.set_trace()
# print("raw image",raw_img.size())
raw_img=torch.rand(1,3,224,224).to(DEVICE)
import pdb;pdb.set_trace()
depth = model(raw_img) # HxW raw depth map in numpy
import pdb;pdb.set_trace()
# raw_img=torch.from_numpy(raw_img)
# raw_img=raw_img.permute(2,0,1).unsqueeze(0).to('cuda')
# raw_img = resize(raw_img).to('cuda').to(torch.float32)
# import pdb;pdb.set_trace()
# # raw_img=torch.rand(8,3,336,336).to('cuda')
# # raw_img=raw_img.numpy()
# depth=model(raw_img)
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

# 将其转换为 8 位无符号整型，因为图像格式通常使用 uint8
depth_normalized = depth_normalized.astype(np.uint8)
cv2.imwrite('depth-a5.jpg', depth_normalized)

# import pdb;pdb.set_trace()
# print("depth",depth.size())
# plt.imshow(depth, cmap='viridis')  # 使用 'viridis' 颜色映射
# plt.title('Depth Map')
# plt.savefig('depth-a1.png')  # 保存图像到文件