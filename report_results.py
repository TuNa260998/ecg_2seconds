
from model import seresnet
from model import resnet
from utils import static
import os
import torch
from utils import test

def table_report(path,args,val_loader,criterion,device):
    data=[]
    for file in os.listdir(args.path):
        if file=='resnet18.ptl':
            net = resnet.resnet18(input_channels=12).to(device)
        elif file=='resnet34.ptl':
            net = resnet.resnet34(input_channels=12).to(device)
        elif file=='se_resnet18.ptl':
            net = seresnet.se_resnet18().to(device)
        
            
        net.load_state_dict(torch.load(args.path+file, map_location=device))
        size= static.get_size(args.path+file ,'mb')
        
        start = 1
        f1_arr, f1_mean=test.test(val_loader, net, args, criterion, device)
        stop = 2
        time=stop - start
        data.append([file.split(".")[0],f1_mean,size,time,f1_arr[0],f1_arr[1], f1_arr[2], f1_arr[3], f1_arr[4], f1_arr[5], f1_arr[6], f1_arr[7],f1_arr[8]]) 
    col_names = ["Model", "F1_mean",'Size(MB)','Time(s)','F1_SNR', 'F1_AF', 'F1_IAVB', 'F1_LBBB', 'F1_RBBB', 'F1_PAC', 'F1_PVC', 'F1_STD', 'F1_STE']
    return data,col_names