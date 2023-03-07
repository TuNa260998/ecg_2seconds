
import torch
from tqdm import tqdm
import numpy as np
from utils import static
def validation(dataloader, net, args, criterion, device, bestmodel):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
        
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = static.cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)
    
    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        torch.save(net.state_dict(), args.path+bestmodel)
        print("Saved the model")
    else:
        aucs = static.cal_aucs(y_trues, y_scores)
        avg_auc = np.mean(aucs)
        # print('AUCs:', aucs)
        # print('Avg AUC: %.4f' % avg_auc)