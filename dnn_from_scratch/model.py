import numpy as np
from pathlib import Path
from loguru import logger
import pickle

class Model():
    def __init__(self, input_dim, hidden_dim1, out_dim, weight_decay, lr, test_img, test_label):
        self.weight_decay = weight_decay
        self.activate = self.ReLU
        self.params = {
            'W1': np.random.randn(hidden_dim1, input_dim) * np.sqrt(1. / hidden_dim1),
            'b1': np.random.randn(hidden_dim1, 1) * np.sqrt(1. / hidden_dim1),
            'W2': np.random.randn(out_dim, hidden_dim1) * np.sqrt(1. / out_dim),
            'b2': np.random.randn(out_dim, 1) * np.sqrt(1. / out_dim)
        }
        self.lr = lr
        self.save_path = Path('./ckpt/')
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.test_img = test_img
        self.test_label = test_label
        
    def ReLU(self, x):
        return np.maximum(0, x)
    
    def ReLu_d(self, x):
        return (x >= 0).astype(np.float64)
    
    def softmax(self, x):
        exps = np.exp(x - x.max(0))
        return exps / np.sum(exps, axis=0)
    
    def softmax_d(self, x):
        exps = np.exp(x - x.max(0))
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    
    def forward(self, x):
        params = self.params
        params['A0'] = x.T
        params['Z1'] = np.dot(params["W1"], params['A0']) + params['b1']
        params['A1'] = self.ReLU(params['Z1'])
        params['Z2'] = np.dot(params["W2"], params['A1']) + params['b2']
        params['A3'] = self.softmax(params['Z2'])
        return params['A3']
    
    def backward(self, output, y_train, batch_size):
        params = self.params
        change_w = {}
        
        error = (output.T - y_train) / batch_size
        
        change_w['W2'] = np.dot(error.T, params['A1'].T)
        change_w['b2'] = np.sum(error, axis=0, keepdims=True).T
        
        error = np.dot(params['W2'].T, error.T) * self.ReLu_d(params['Z1'])
        change_w['W1'] = np.dot(error, params['A0'].T)
        change_w['b1'] = np.sum(error, axis=1, keepdims=True)
        
        change_w['W1'] += self.weight_decay * params['W1']
        change_w['W2'] += self.weight_decay * params['W2']
        
        return change_w
    
    def cross_entropy(self, pre_y, y):
        predictions = np.clip(pre_y, 1e-12, 1.-1e-12).T
        N = predictions.shape[0]
        ce = - np.sum(y * np.log(predictions)) / N
        return ce
    
    def update_params(self, change_w):
        params = self.params
        for key in change_w:
            params[key] -= self.lr * change_w[key]
    
    def save_model(self):
        params_to_save = {k: self.params[k] for k in ['W1', 'W2', 'b1', 'b2']}
        with open(self.save_path / 'model.pkl', 'wb') as f:
            pickle.dump(params_to_save, f)
        logger.info(f"save model to {self.save_path / 'model.pkl'}")
    
    
    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                loaded_params = pickle.load(f)
                
            for key, value in loaded_params.items():
                self.params[key] = value
                
            logger.info(f"load model from {self.save_path / 'model.pkl'}")
        except Exception as e:
            logger.error(f"load error: {str(e)}")

    def eval(self, test_img, test_label):
        output = self.forward(test_img)
        pred = np.argmax(output, 0)
        gt = np.argmax(test_label, 1)
        acc = np.sum(pred == gt) / len(gt)
        loss = self.cross_entropy(output, test_label)
        return acc, loss