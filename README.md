# FDU-DL-Course
DATA620004 神经网络与深度学习
```
git clone https://github.com/Lexarymade/FDU-DL-Course.git
```
## Homework1 dnn from scratch
```
cd dnn_from_scratch/
```
数据下载: https://www.cs.toronto.edu/~kriz/cifar.html 放到`data/`目录下

模型权重: https://pan.baidu.com/s/12y-jomYjZqb5b-cLO28ZKw?pwd=4faa  放到 `ckpt/`目录下

模型训练与测试运行命令：`python train.py` 如果想直接测试，需要在文件中给出`model_path`并指定`is_train=False`再运行命令即可


## Final Project 3DGS
```
cd gaussian-splatting/
```
### 环境配置：
- COLMAP：follow [colmap配置](https://blog.csdn.net/Sakuya__/article/details/134766215)
```
# 1. 上面的colmap的编译步骤存在问题
apt-get update
apt-get install -y build-essential cmake ninja-build

# 2. 确认CUDA/NVCC正常
/usr/local/cuda/bin/nvcc -V

# 3. 重新进到build目录，清理残留
cd ~/autodl-tmp/gaussian-splatting/colmap
rm -rf build && mkdir build && cd build

# 4. 重新运行CMake
cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86

# 5. 编译
ninja -j$(nproc)

# 6. 测试
ninja install          # ninja install 会把colmap复制到 /usr/local/bin/，所以所有终端都能直接调用
colmap help                 
```
- 3DGS：follow [知乎链接1](https://zhuanlan.zhihu.com/p/1889024280211199152)、[知乎链接2](https://zhuanlan.zhihu.com/p/10133731526)、[csdn文章](https://blog.csdn.net/Sakuya__/article/details/135376331)
- Viewer: follow [nerfstudio](https://github.com/yzslab/nerfstudio/tree/gaussian_splatting)

### 代码使用：
- Train:
```
# 在这个bash文件中配置好数据与output dir
bash train_video.sh
```
- Viewer:
```
cd ../nerfstudio/
python nerfstudio/scripts/gaussian_splatting/run_viewer.py --model-path GAUSSIAN_TRAINING_OUTPUT_MODEL_DIR
```

- Render with custom camera pose:
```
cd ../nerfstudio/
python nerfstudio/scripts/gaussian_splatting/render.py camera-path \
    --model-path GAUSSIAN_TRAINING_OUTPUT_MODEL_DIR \
    --camera-path-filename YOUR_CAMERA_PATH_FILE.json \
    --output-path YOUR_OUTPUT_MP4_FILE.mp4
```



