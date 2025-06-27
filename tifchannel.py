import bioformats
import numpy as np
import os

folder_path = 'your_folder_path'  # 修改为你的文件夹路径
files = [f for f in os.listdir(folder_path) if not f.endswith('.tif')]

for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    
    try:
        # 使用pybioformats读取无扩展名的文件
        with bioformats.ImageReader(file_path) as reader:
            img_data = reader.read(rescale=False)  # 读取图像数据
            
            # 假设数据是4D (time, channel, height, width)
            channel_1_data = img_data[:, 0, :, :]  # 提取第一个通道
            
            # 保存为新的单通道文件
            new_file_path = os.path.join(folder_path, f'channel1_{file_name}.tif')
            bioformats.write_image(new_file_path, channel_1_data)
        
        print(f"保存新文件: {new_file_path}")
    
    except Exception as e:
        print(f"无法处理文件 {file_name}: {e}")
