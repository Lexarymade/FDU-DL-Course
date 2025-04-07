import numpy as np
import matplotlib.pyplot as plt
from model import Model
from dataloader import DataLoader
from loguru import logger

def visualize_weights(model_weights):
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.title('W1 Weights')
    plt.imshow(model_weights['W1'], cmap='viridis')
    plt.colorbar()
    
    plt.subplot(122)
    plt.title('W2 Weights')
    plt.imshow(model_weights['W2'], cmap='viridis')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f'./weights_visualization.png')
    plt.close()
    logger.info(f"权重矩阵可视化已保存")


if __name__ == '__main__':
    model = Model(
        input_dim=3072,  # CIFAR-10: 32x32x3
        hidden_dim1=2048,
        out_dim=10,
        weight_decay=0.0001,
        lr=0.001,
        test_img=None,
        test_label=None
    )
    
    # load model
    model.load_model("/root/autodl-tmp/model.pkl")
    visualize_weights(model.params)