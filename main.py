from libs import *
from model import resnet
from utils import ECG_DB
from utils import train
from utils import validation
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
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=[1,2], action='store_true', help='Use GPU')
    
    return parser.parse_args()


if __name__== "__main__":
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    bestmodel="bestmodel.ptl"

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    
    label_csv = os.path.join(data_dir, 'labels.csv')
    
    train_folds, val_folds, test_folds = static.split_data(seed=args.seed)
    
    train_dataset = ECG_DB.ECG_DB('train', data_dir, label_csv, train_folds)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = ECG_DB.ECG_DB('val', data_dir, label_csv, val_folds)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataset = ECG_DB.ECG_DB('test', data_dir, label_csv, test_folds)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    net = resnet.resnet34(input_channels=nleads).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()
    
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            train.train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            validation.validation(val_loader, net, args, criterion, device, bestmodel)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        validation.validation(test_loader, net, args, criterion, device)