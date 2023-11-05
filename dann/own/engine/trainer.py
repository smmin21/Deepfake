import time
import os
import torch
import torch.nn as nn
import os.path as osp
import pdb
import matplotlib.pyplot as plt
import numpy as np

class Trainer():
    def __init__(self, opt, data_loader, model, logger):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = data_loader['train']
        self.val_loader = data_loader['val']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.TRAIN.lr)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        # self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=50, gamma=0.25)

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
        total_steps = 0
        err_label_list = []
        err_domain_list = []
        print("Start training,,,")
        for epoch in range(self.total_epoch):
            self.model.train()
            steps = 0
            err_label_sum = 0
            err_domain_sum = 0
            for data in self.train_loader: # len(self.train_loader) = train data size / batch size  # len(self.train_loader.dataset) = train data size
                p = float(steps + epoch * len(self.train_loader)) / self.total_epoch / len(self.train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                # print('current step: ', steps)
                steps += 1
                total_steps += 1
                # run step
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                class_output, domain_output = self.model(data['frame'].to(self.device), alpha)
                
                domain_mask = data["domain_label"] != 4
                # num_real = torch.sum(data["domain_label"] == 4)//4
                # domain_mask = data["domain_label"] != 4
                # cnt = 0
                # for idx, boolean in enumerate(domain_mask):
                #     if (not boolean) and (cnt<num_real):
                #         domain_mask[idx] = not boolean
                #         cnt += 1
                
                err_label = self.loss_function(class_output, data['label'].to(self.device))
                err_domain = self.loss_function(domain_output[domain_mask, :], data['domain_label'][domain_mask].to(self.device))
                train_loss = err_label + err_domain
                train_loss.backward()
                if self.optimizer is not None:
                    self.optimizer.step()
                err_label_sum += err_label
                err_domain_sum += err_domain
                if self.logger is not None:
                    if steps%self.log_interval == 0:
                        self.logger.info(f"err_label: {err_label_sum.item()/steps:>7f}  err_domain: {err_domain_sum.item()/steps:>7f}  [{steps:>5d}/{len(self.train_loader):>5d}]")
    
                if total_steps%self.save_interval == 0:
                    self.save_model(total_steps, epoch)
            # self.scheduler.step()
                    
            err_label_list.append(err_label_sum/len(self.train_loader))
            err_domain_list.append(err_domain_sum/len(self.train_loader))
            total, correct, val_loss = self.validate()
            self.logger.info('Epoch: %d/%d, Train loss: %.6f, val loss: %.6f, Accuracy: %.2f' %(epoch+1, self.total_epoch, train_loss/len(self.train_loader), val_loss, 100*correct/total))
        self.save_model(total_steps, epoch)
        self.save_train_loss_graph(err_label_list, 'label')
        self.save_train_loss_graph(err_domain_list, 'domain')
        self.logger.info('Finished Training : total steps %d' %total_steps)

    def validate(self):
        total, correct, val_loss = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                class_output, _ = self.model(data['frame'].to(self.device), 0)
                
                _, predicted = torch.max(class_output.data, 1)
                # total += data['label'].to(self.device).size(0)
                total += data['label'].size(0)
                correct += (predicted == data['label'].to(self.device)).sum().item()
                val_loss += self.loss_function(class_output, data['label'].to(self.device)).item()
        return total, correct, val_loss/len(self.val_loader)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.load_ckpt_dir))
            
    def save_model(self, steps, epoch):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), osp.join(self.ckpt_dir, f'step{steps}_ep{epoch+1}.pt'))

    def save_train_loss_graph(self, train_loss_list, type):   
        epochs = [i+1 for i in range(self.total_epoch)]   
        if not isinstance(train_loss_list, list):
            train_loss_list = [train_loss_list]
        train_loss_list = [loss.cpu().detach().numpy() for loss in train_loss_list]
        plt.plot(epochs, train_loss_list, label='Train Loss')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        plt.savefig(osp.join(self.ckpt_dir, 'train_{}_loss.png'.format(type)))
        plt.close()
