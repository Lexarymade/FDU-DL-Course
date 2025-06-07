import wandb
from loguru import logger
import itertools
from dataloader import DataLoader
from train import train_model, Model
import yaml
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split


def load_config(config_path: str = 'hp_config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"配置文件加载失败: {str(e)}")
        return 
        
    
def load_and_preprocess_data(data_path: str) -> Tuple[Any, Any, Any, Any, Any, Any]:

    loader = DataLoader(data_path)
    train_img, train_label, test_img, test_label = loader.load_data()
    
    train_img, val_img, train_label, val_label = train_test_split(
        train_img, train_label, test_size=0.2, random_state=42
    )
    
    train_img, val_img, test_img = loader.preprocess_data(train_img, val_img, test_img)
    
    return train_img, train_label, val_img, val_label, test_img, test_label


def train_single_configuration(
    params: tuple,
    train_img: Any,
    train_label: Any,
    val_img: Any,
    val_label: Any,
    test_img: Any,
    test_label: Any,
    project_name: str,
    train_params: Dict[str, Any]
) -> None:

    hidden_dim1, lr, weight_decay = params
    run_name = f'hidden_dim1_{hidden_dim1}_lr_{lr}_weight_decay_{weight_decay}'
    
    try:
        with wandb.init(project=project_name, name=run_name, reinit=True) as run:
            logger.info(f'开始训练: {run_name}')
            
            model = Model(
                input_dim=3072,
                hidden_dim1=hidden_dim1,
                out_dim=10,
                weight_decay=weight_decay,
                lr=lr,
                test_img=val_img,  # 使用验证集进行训练过程中的评估
                test_label=val_label
            )
            
            train_model(
                model=model,
                train_img=train_img,
                train_label=train_label,
                val_img=val_img,
                val_label=val_label,
                batch_size=train_params['batch_size'],
                epoch_size=train_params['epoch_size'],
                enable_wandb=True
            )
            
            # 在测试集上进行最终评估
            test_acc, test_loss = model.eval(test_img, test_label)
            logger.info(f"Final test results - accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
            
            wandb.log({
                "test_accuracy": test_acc,
                "test_loss": test_loss
            })
            
    except Exception as e:
        logger.error(f"模型训练失败 {run_name}: {str(e)}")


def run_hyperparameter_tuning():
    
    config = load_config()
    train_img, train_label, val_img, val_label, test_img, test_label = load_and_preprocess_data(
        config['data_path']
    )
    
    model_params = config['model_params']
    param_combinations = itertools.product(
        model_params['hidden_dim1'],
        model_params['lr'],
        model_params['weight_decay']
    )
    
    for params in param_combinations:
        train_single_configuration(
            params=params,
            train_img=train_img,
            train_label=train_label,
            val_img=val_img,
            val_label=val_label,
            test_img=test_img,
            test_label=test_label,
            project_name=config['project_name'],
            train_params=config['train_params']
        )

if __name__ == '__main__':
    run_hyperparameter_tuning()