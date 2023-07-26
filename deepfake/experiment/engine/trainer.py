import time

import torch
import torch.nn as nn
import os.path as osp
import pdb
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
        self.steps = 0
        
    def train(self):
        print('start training')
        for epoch in range(self.total_epoch):
            self.model.train()
            for data in self.train_loader:
                # print('current step: ', self.steps)
                self.steps += 1
                # run step
                train_loss = self.run_step(data)

                if self.logger is not None:
                    if self.steps%self.log_interval == 0:
                        train_loss = train_loss.item()
                        self.logger.info(f"loss: {train_loss:>7f}  [{self.steps:>5d}/{len(self.train_loader.dataset):>5d}]")
    
                if self.steps%self.save_interval == 0:
                    self.save_model(epoch)

            total, correct, val_loss = self.validate()
            self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f, Accuracy: %.2f' %(epoch+1, self.total_epoch, train_loss, val_loss, 100*correct/total))

    def validate(self):
        total, correct, val_loss = 0, 0, 0
        accuracy = []
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                output = self.model(data['frame'].to(self.device))
                
                _, predicted = torch.max(output.data, 1)
                total += data['label'].to(self.device).size(0)
                correct += (predicted == data['label'].to(self.device)).sum().item()
                val_loss = self.loss_function(output, data['label'].to(self.device)).item()
                accuracy.append(100 * correct/total)
        return total, correct, val_loss

    def run_step(self, data):
        # forward / loss / backward / update
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        start_time = time.time()
        output = self.model(data['frame'].to(self.device)).logits
        end_time = time.time()
        # print(f'forward time: {end_time-start_time}')
        train_loss = self.loss_function(output, data['label'].to(self.device))
        train_loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
        return train_loss

    def load_model(self):
        pass
    
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), osp.join(self.ckpt_dir, f'step{self.steps}_ep{epoch+1}.pt'))

