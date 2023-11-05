import time

import torch
import torch.nn as nn
import os.path as osp
import pdb
from sklearn.metrics import roc_auc_score

class Tester():
    def __init__(self, opt, data_loader, model, logger):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = data_loader['train']
        self.val_loader = data_loader['val']
        self.test_loader = data_loader['test']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.TRAIN.lr)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        
    def test(self, split="test"):
        total, correct, loss = 0, 0, 0
        accuracy = []
        self.model.eval()
        if split == "train":
            dataloader = self.train_loader
        elif split == "test":
            dataloader = self.test_loader
        else:
            dataloader = self.val_loader
        with torch.no_grad():
            pred = []
            true = []
            for data in dataloader:
                output = self.model(data['frame'].to(self.device))
                _, predicted = torch.max(output.data, 1)
                
                pred += output.data[:, 1].cpu().tolist()
                true += data['label'].cpu()
                
                total += data['label'].to(self.device).size(0)
                correct += (predicted == data['label'].to(self.device)).sum().item()
                loss = self.loss_function(output, data['label'].to(self.device)).item()
                accuracy.append(100 * correct/total)
            auc_score = roc_auc_score(true, pred) * 100
        self.logger.info('[ %s result ] loss: %.6f, Accuracy: %.2f, AUC: %.2f' %(split, loss, 100*correct/total, auc_score))
        # return total, correct, val_loss