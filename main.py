from libs import *
from model import resnet
from utils import ECG_DB
from utils import train
from utils import validation
from utils import static
from model import seresnet
import report_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/chapman/CPSC/', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=[1,2], action='store_true', help='Use GPU')
    parser.add_argument('--path', type=str, default="/home/ubuntu/tu.na/ecg_2seconds/result/")
    
    return parser.parse_args()



if __name__== "__main__":
    args = parse_args()
    args.best_metric = 0
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:2')
    data_dir = os.path.normpath(args.data_dir)
    
    bestmodel="se_resnext18.ptl"
    
    net = seresnet.se_resnext18().to(device)
    
    
    label_csv = os.path.join(data_dir, 'labels.csv')
    
    train_folds=[ 9 , 2 , 6 , 1 , 8,  3 ,10 , 5]
    val_folds=[4,7]
    
    train_dataset = ECG_DB.ECG_DB('train', data_dir, label_csv, train_folds)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_dataset = ECG_DB.ECG_DB('val', data_dir, label_csv, val_folds)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()
    
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.path, map_location=device))
            
        for epoch in range(args.epochs):
            train.train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            validation.validation(val_loader, net, args, criterion, device, bestmodel)
    else:
        data, col_names=report_results.table_report(args.path,args,val_loader,criterion,device)
        print(tabulate(data, headers=col_names, tablefmt="fancy_grid", showindex="always"))
        
        