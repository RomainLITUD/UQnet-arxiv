import torch
#from torch.cuda.amp import autocast, GradScaler
from heatmap_model.utils import *
from heatmap_model.interaction_dataset import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(epoch_index, batch_size, model, optimizer, loss_2, training_loader, scheduler, mode):

    running_loss = 0.
    last_loss = 0.
    
    if mode=="lanescore":
        for j, data in enumerate(training_loader):
            traj, splines, masker, lanefeature, adj, af, ar, c_mask, y, ls = data
            optimizer.zero_grad()
            lsp, heatmap, heatmap_reg = model(traj, splines, masker, lanefeature, adj, af, ar, c_mask)
            loss = loss_2([lsp, heatmap, heatmap_reg], [ls, y])
            print('loss:'+str(loss.item()),end='\r')
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if j % 800 == 799:
                scheduler.step()
                last_loss = running_loss / 800
                print('  batch {} loss: {}'.format(j + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + j + 1
                running_loss = 0.
    
    if mode=="testmodel":
        for j, data in enumerate(training_loader):
            traj, splines, masker, lanefeature, adj, af, al, c_mask, y = data
            optimizer.zero_grad()
            heatmap = model(traj, splines, masker, lanefeature, adj, af, al, c_mask)
            #heatmap2 = model2(traj, splines, masker, lanefeature, adj, af, al)
            loss = loss_2(heatmap, y)
            #loss2 = loss_2(heatmap2, y)
            #loss = loss1 + loss2
            print('loss:'+str(loss.item()),end='\r')
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if j % 800 == 799:
                scheduler.step()
                last_loss = running_loss / 800
                print('  batch {} loss: {}'.format(j + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + j + 1
                running_loss = 0.
    
    if mode=="traj":
        for j, data in enumerate(training_loader):
            x, y = data
            optimizer.zero_grad()
            yp = model(x, y)
            loss = loss_2(yp, y)
            print('loss:'+str(loss.item()),end='\r')
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if j % 800 == 799:
                scheduler.step()
                last_loss = running_loss / 800
                print('  batch {} loss: {}'.format(j + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + j + 1
                running_loss = 0.
                
    return last_loss
        
def train_model(epochs, batch_size, trainset, model, optimizer, validation_loader, loss_2, scheduler, para, mode):
    training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    epoch_number = 0

    EPOCHS = epochs

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        #model.half()
        avg_loss = train_one_epoch(epoch_number, batch_size, model, optimizer, loss_2, training_loader, scheduler, mode)
        model.train(False)

        running_vloss = 0.0
        if mode=="lanescore":
            for i, vdata in enumerate(validation_loader):
                traj, splines, masker, lanefeature, adj, af, ar, c_mask, y, ls = vdata
                lsp, heatmap, heatmap_reg = model(traj, splines, masker, lanefeature, adj, af, ar, c_mask)
                heatmaploss = loss_2([lsp, heatmap, heatmap_reg], [ls, y])
                running_vloss += float(heatmaploss)
                
        if mode=="testmodel":
            for i, vdata in enumerate(validation_loader):
                traj, splines, masker, lanefeature, adj, af, al, c_mask, y = vdata
                heatmap = model(traj, splines, masker, lanefeature, adj, af, al, c_mask)
                heatmaploss = loss_2(heatmap, y)
                running_vloss += float(heatmaploss)
                
        if mode=="traj":
            for i, vdata in enumerate(validation_loader):
                x, y = vdata
                yp = model(x,y)
                loss = loss_2(yp, y)
                running_vloss += float(loss)

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        scheduler.step()
        epoch_number += 1