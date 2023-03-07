from libs import *
from model import resnet
from utils import ECG_DB
from utils import train
from utils import validation
from utils import test
from utils import static



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/chapman/CPSC/', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=[1,2], action='store_true', help='Use GPU')
    parser.add_argument('--path', type=str, default="/home/ubuntu/tu.na/ecg_2seconds/result/")
    
    return parser.parse_args()


if __name__== "__main__":
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    bestmodel="bestmodel1.ptl"

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:2')
    
    
    label_csv = os.path.join(data_dir, 'labels.csv')
    
    train_folds, val_folds, test_folds = static.split_data(seed=args.seed)
    
    train_dataset = ECG_DB.ECG_DB('train', data_dir, label_csv, train_folds)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = ECG_DB.ECG_DB('val', data_dir, label_csv, val_folds)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataset = ECG_DB.ECG_DB('test', data_dir, label_csv, test_folds)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    net = resnet.resnet34(input_channels=12).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()
    
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.path, map_location=device))
            
        for epoch in range(args.epochs):
            train.train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            validation.validation(val_loader, net, args, criterion, device, bestmodel)
    else:
        data=[]
    
        for file in os.listdir(args.path):   
            net.load_state_dict(torch.load(args.path+file, map_location=device))
            f1_arr, f1_mean=test.test(test_loader, net, args, criterion, device)
            data.append(["Resnet34",f1_mean,f1_arr[0],f1_arr[1], f1_arr[2], f1_arr[3], f1_arr[4], f1_arr[5], f1_arr[6], f1_arr[7],f1_arr[8]])   

        col_names = ["Model", "F1_mean",'F1_SNR', 'F1_AF', 'F1_IAVB', 'F1_LBBB', 'F1_RBBB', 'F1_PAC', 'F1_PVC', 'F1_STD', 'F1_STE']
        print(tabulate(data, headers=col_names, tablefmt="fancy_grid", showindex="always"))
        
        