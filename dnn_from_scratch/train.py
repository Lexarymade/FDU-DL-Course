import numpy as np
from dataloader import DataLoader
from tqdm import tqdm
import wandb
from loguru import logger
from model import Model
from sklearn.model_selection import train_test_split

def train_model(model, train_img, train_label, val_img, val_label, batch_size=32, epoch_size=10, enable_wandb=False):
    
    best_acc = 0
    dataset_length = len(train_img)
    iteration_size = dataset_length // batch_size
    
    for epoch in range(epoch_size):
        epoch_loss = 0
        cnt = 0
        
        for k in tqdm(range(iteration_size), desc=f"Epoch {epoch+1}/{epoch_size}"):
            
            # get training data
            batch_index = np.random.choice(range(dataset_length), batch_size, replace=False)
            batch_img, batch_label = train_img[batch_index], train_label[batch_index]
            
            # forward
            output = model.forward(batch_img)
            loss = model.cross_entropy(output, batch_label)
            epoch_loss += loss
            
            # backward
            change_w = model.backward(output, batch_label, batch_size)
            
            # calculate accuracy
            cnt += np.sum(np.argmax(batch_label, 1) == np.argmax(output, 0))
            
            # update params
            model.update_params(change_w)
            
            # log batch loss
            if enable_wandb:
                wandb.log({"batch_loss": loss})
        
        train_acc = cnt / dataset_length
        val_acc, val_loss = model.eval(test_img=val_img, test_label=val_label)
        
        # log metrics
        if enable_wandb:
            wandb.log({
                "train_loss": epoch_loss / iteration_size,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc
            })
        logger.info(
            f'epoch: {epoch+1} train_loss = {epoch_loss/iteration_size:.4f} '
            f'val_loss = {val_loss:.4f} train_acc = {train_acc:.4f} '
            f'val_acc = {val_acc:.4f}'
        )

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_model()
            logger.info(f"save best model, val_acc: {best_acc:.4f}")

if __name__ == '__main__':

    enable_wandb = True
    if enable_wandb:
        wandb.init(project="dnn_from_scratch",name="baseline")  

    dataloader = DataLoader('./data')
    train_img, train_label, test_img, test_label = dataloader.load_data()
    
    # 从训练集中分割出验证集
    train_img, val_img, train_label, val_label = train_test_split(
        train_img, train_label, test_size=0.2, random_state=42
    )
    
    # 
    train_img, val_img, test_img = dataloader.preprocess_data(train_img, val_img, test_img)
    
    model = Model(
        input_dim=3072,  # CIFAR-10: 32x32x3=3072
        hidden_dim1=2048,
        out_dim=10,
        weight_decay=0.0001,
        lr=0.01,
        test_img=val_img, 
        test_label=val_label,
    )
    
    is_train = True
    if is_train:
        train_model(
            model, 
            train_img, 
            train_label, 
            val_img, 
            val_label,
            batch_size=64, 
            epoch_size=10, 
            enable_wandb=enable_wandb
        )
    
        # test
        test_acc, test_loss = model.eval(test_img, test_label)
        logger.info(f"Final test results - accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
    
    else:
        model_path = 'your_model_path'
        model.load_model(model_path)
        test_acc, test_loss = model.eval(test_img, test_label)
        logger.info(f"Final test results - accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")