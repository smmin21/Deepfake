import time
import os
import torch
import torch.nn as nn
import os.path as osp
import pdb
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, opt, data_loader, model, logger):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.train_loader = data_loader['train']
        self.val_loader = data_loader['val']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.TRAIN.lr)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.logger = logger

        self.total_epoch = opt.TRAIN.epochs
        self.log_interval = opt.TRAIN.log_interval
        self.save_interval = opt.TRAIN.save_interval
        self.ckpt_dir = opt.TRAIN.ckpt_dir
        self.load_ckpt_dir = opt.TRAIN.load_ckpt_dir
        
        
    def train(self):
        if self.load_ckpt_dir != 'None':
            self.load_model()
            print('load model from ', self.load_ckpt_dir)
            return
        else:
            print('no ckpt to load!')
        print('start training')
        total_steps = 0
        train_loss_list = []
        for epoch in range(self.total_epoch):
            self.model.train()
            steps = 0
            train_loss = 0
            for data in self.train_loader: # len(self.train_loader) = train data size / batch size  # len(self.train_loader.dataset) = train data size
                # print('current step: ', steps)
                steps += 1
                total_steps += 1
                # run step
                train_loss += self.run_step(data) 
                if self.logger is not None:
                    if steps%self.log_interval == 0:
                        self.logger.info(f"loss: {train_loss.item()/steps:>7f}  [{steps:>5d}/{len(self.train_loader):>5d}]")
    
                if total_steps%self.save_interval == 0:
                    self.save_model(total_steps, epoch)
                    
            train_loss_list.append(train_loss/len(self.train_loader))
            total, correct, val_loss = self.validate()
            self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f, Accuracy: %.2f' %(epoch+1, self.total_epoch, train_loss/len(self.train_loader), val_loss, 100*correct/total))
        self.save_model(total_steps, epoch)
        self.save_train_loss_graph(train_loss_list)
        self.logger.info('Finished Training : total steps %d' %total_steps)

    def validate(self):
        total, correct, val_loss = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                output = self.model(data['frame'].to(self.device))
                
                _, predicted = torch.max(output.data, 1)
                # total += data['label'].to(self.device).size(0)
                total += data['label'].size(0)
                correct += (predicted == data['label'].to(self.device)).sum().item()
                val_loss += self.loss_function(output, data['label'].to(self.device)).item()
        return total, correct, val_loss/len(self.val_loader)

    def run_step(self, data):
        # forward / loss / backward / update
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        output = self.model(data['frame'].to(self.device)).logits
        train_loss = self.loss_function(output, data['label'].to(self.device))
        train_loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
        return train_loss

    def load_model(self):
        self.model.load_state_dict(torch.load(self.load_ckpt_dir))
            
    def save_model(self, steps, epoch):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), osp.join(self.ckpt_dir, f'step{steps}_ep{epoch+1}.pt'))

    def save_train_loss_graph(self, train_loss_list):   
        epochs = [i+1 for i in range(self.total_epoch)]     
        plt.plot(epochs, train_loss_list, label='Train Loss')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        plt.savefig(osp.join(self.ckpt_dir, 'train_loss.png'))
        plt.close()
