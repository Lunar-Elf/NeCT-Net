import torch
import math
import torch.utils.data as data
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm


def train(args, Train_loader, Val_loader, net, optimizer, criterion):
    net.to(args.device)
    '''
    Start training
    '''


    for e in tqdm(range(1, args.num_epochs + 1), desc="Training the network"):
        net.train()
        train_loss = 0.0
        # print(len(train_loader))
        for index, (data, target) in enumerate(Train_loader):
            data, target = data.to(args.device), target.to(args.device)

            data = data

            optimizer.zero_grad()
            output = net(data)

            loss = criterion(output.squeeze().float(), target.float())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(Train_loader)
        print("epoch:", (e), " avg_train_loss:", (avg_train_loss))
        '''
        Start validation
        '''
        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            up = 0.0
            real = []
            re = []
            down = 0.0
            for batch_idx, (data, target) in enumerate(Val_loader):
                data, target = data.to(args.device), target.to(args.device)
                data = data
                output = net(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                # print(target,output)

                output = output.detach().cpu().numpy()[0][0]
                # print(output)
                target = target.cpu().numpy()[0]

                up += (output - target) * (output - target)
                real.append(target)

                re.append(abs((target - output) / (target + 0.0000001)))
            for i in range(len(real)):
                down += (np.mean(real) - real[i]) * (np.mean(real) - real[i])
            avg_val_loss = val_loss / len(Val_loader)
            r2 = 1 - up / down
            mse = up / len(real)
            rmse = math.sqrt(mse)
            mre = np.mean(re)
            print(rmse)
            torch.save(net, "./model/epoch=" + str(e) + "train_loss="
                       + str(round(avg_train_loss, 4)) + "val_loss=" + str(round(avg_val_loss, 4))
                       + "r2=" + str(round(r2, 4)) + "MRE=" + str(round(mre, 4)) +
                       "MSE=" + str(round(mse, 4)) + "RMSE=" + str(round(rmse, 4)) + ".pkl")





