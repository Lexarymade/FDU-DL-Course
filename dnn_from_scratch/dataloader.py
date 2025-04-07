from pathlib import Path
import numpy as np
import os, pickle

class DataLoader():
    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)
        self.cifar_folder = "cifar-10-batches-py"
        self.processed_data = self.data_dir / 'cifar10_data.pkl'
        
    def load_cifar_batch(self, batch_file):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')
            
        images = batch_data[b'data']
        labels = np.array(batch_data[b'labels'])
        label_one_hot = np.zeros((labels.shape[0], 10))
        for i, label in enumerate(labels):
            label_one_hot[i, label] = 1
            
        return images, label_one_hot, labels
    
    def load_data(self):
        if not os.path.exists(self.processed_data):
            extract_dir = self.data_dir / self.cifar_folder 
            if not extract_dir.exists():
                raise FileNotFoundError(f"未找到CIFAR-10数据目录: {extract_dir}")
            
            train_images = []
            train_labels = []
            train_raw_labels = []
            
            for i in range(1, 6):
                batch_file = extract_dir / f"data_batch_{i}"
                if not batch_file.exists():
                    raise FileNotFoundError(f"未找到批次文件: {batch_file}")
                    
                images, labels, raw_labels = self.load_cifar_batch(batch_file)
                train_images.append(images)
                train_labels.append(labels)
                train_raw_labels.append(raw_labels)
            
            # merge all training batches
            train_images = np.concatenate(train_images)
            train_labels = np.concatenate(train_labels)
            
            # load test data
            test_batch_file = extract_dir / "test_batch"
            if not test_batch_file.exists():
                raise FileNotFoundError(f"未找到测试批次文件: {test_batch_file}")
                
            test_images, test_labels, _ = self.load_cifar_batch(test_batch_file)
            
            # save processed data
            with open(self.processed_data, 'wb') as f:
                pickle.dump((train_images, train_labels, test_images, test_labels), f)
            
            print(f"processed data saved to {self.processed_data}")
        else:
            # load processed data from cache
            print(f"load processed data from {self.processed_data}")
            with open(self.processed_data, 'rb') as f:
                train_images, train_labels, test_images, test_labels = pickle.load(f)
        
        return train_images, train_labels, test_images, test_labels
    
    def preprocess_data(self, train_images, val_images, test_images):
        """
        数据预处理函数
        Args:
            train_images: 训练集图像
            val_images: 验证集图像
            test_images: 测试集图像
        Returns:
            处理后的训练集、验证集和测试集图像
        """
        # 数据归一化
        train_images = train_images.astype('float32') / 255.0
        val_images = val_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        
        # 使用训练集的统计数据进行标准化
        mean = np.mean(train_images, axis=0)
        std = np.std(train_images, axis=0)
        
        train_images = (train_images - mean) / (std + 1e-7)
        val_images = (val_images - mean) / (std + 1e-7)
        test_images = (test_images - mean) / (std + 1e-7)
        
        return train_images, val_images, test_images

if __name__ == "__main__":
    loader = DataLoader('./data')
    train_images, train_labels, test_images, test_labels = loader.load_data()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)