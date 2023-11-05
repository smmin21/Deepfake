import time

import torch
import torch.nn as nn
import os.path as osp
import pdb

## Image 모델, Video 인풋 데이터(연속 이미지)에 대한 Tester
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
            for data in dataloader: # data : each video
                outputs = []
                for each_frame in data['frame']: # each_frame : each frame
                    output = self.model(each_frame.to(self.device))
                    outputs.append(output)
                outputs = torch.stack(outputs)
                predicted = torch.mode(outputs.argmax(dim=2), dim=0).values
                total += data['label'].to(self.device).size(0)
                correct += (predicted == data['label'].to(self.device)).sum().item()
                ## 손실함수에 outputs가 들어가도 되는건가?
                loss = self.loss_function(outputs, data['label'].to(self.device)).item()
                accuracy.append(100 * correct/total)
        self.logger.info('[ %s result ] loss: %.6f, Accuracy: %.2f' %(split, loss, 100*correct/total))
        # return total, correct, val_loss