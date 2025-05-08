from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import dataset.celeba as dataset
import models
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.ema import ModelEMA
from tensorboardX import SummaryWriter
from utils import AverageMeter, Bar, Logger, mkdir_p, savefig
import matplotlib.pyplot as plt
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize (default: 32)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                     metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', type=str, default='step',
                     help='mode for learning rated ecay')
parser.add_argument('--step', type=int, default=10,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./outputs', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--cardinality', type=int, default=32, help='ResNeXt model cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNeXt model base width (number of channels in each group).')
parser.add_argument('--groups', type=int, default=3, help='ShuffleNet model groups')
# Miscs
parser.add_argument('--manual-seed', type=int, default=1, 
                        help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
#Device options
parser.add_argument('--gpu-id', default='0,1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--label-ratio', type=float, default=0.02,
                        help='ratio of labeled data')
parser.add_argument('--mu', default=7, type=int,
                        help='relative sizes of labeled batch and unlabeled batch')
parser.add_argument('--boundary-value', default=0.5, type=float,
                        help='boundary value')
parser.add_argument('--threshold0', default=0.5, type=float,
                        help='pseudo label threshold for the presence of attributes')
parser.add_argument('--threshold1', default=0.5, type=float,
                        help='pseudo label threshold for the absence of attributes')
parser.add_argument('--lambda-u', default=1, type=float,
                        help='couse_ratecient of unlabeled loss')
parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
#Data
parser.add_argument('-d', '--data', default='./datas', type=str)
parser.add_argument('--label-train', default='./datas/train_40_att_list.txt', type=str, help='train label file')
parser.add_argument('--label-test', default='./datas/test_40_att_list.txt', type=str, help='test label file')
parser.add_argument('--train-data', default=162770, type=int, metavar='N',
                    help='number of train data (default: 162770)')
parser.add_argument('--test-data', default=19962, type=int, metavar='N',
                    help='number of test data (default: 19962)')


best_prec1 = 0
torch.set_printoptions(threshold = np.inf)


def main():   
    global args, best_prec1
    args = parser.parse_args()

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)
        
    # Create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained = True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    
    model = torch.nn.DataParallel(model).cuda()
   
    # Define loss function and optimizer
    criterion = nn.BCELoss(reduction = 'none')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    
    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)

    # Optionally resume from a checkpoint
    title = 'CelebA-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume): 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = torch.device("cpu"))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.use_ema:
                ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title = title, resume = True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))  
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title = title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Loss_x', 'Loss_u','Test Loss', 'Use Rate', 'Test Acc'])

    cudnn.benchmark = True 

    # Data loading
    train_labeled_dataset, train_unlabeled_dataset, test_dataset = dataset.get_celeba(
                                    args.data, args.label_train, args.label_test, args.label_ratio)
        
    labeled_trainloader = torch.utils.data.DataLoader(
        train_labeled_dataset,
        batch_size = args.train_batch,  
        shuffle = True,
        num_workers = args.workers,
        pin_memory = False,
        drop_last = True
        )
    
    unlabeled_trainloader = torch.utils.data.DataLoader(
        train_unlabeled_dataset,
        batch_size = args.train_batch*args.mu,      
        shuffle = True,
        num_workers = args.workers,
        pin_memory = False,
        drop_last = True
        )  

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = args.test_batch, 
        shuffle=False,
        num_workers = args.workers, 
        pin_memory=False
        ) 

    if args.evaluate:
        test(test_loader, model, criterion) 
        return

    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    threshold0 = [args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0, args.threshold0]
    threshold1 = [args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1, args.threshold1]
    
    for epoch in range(args.start_epoch, args.epochs): 
        lr = adjust_learning_rate(optimizer, epoch)
        optimizer.step()  
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs,  lr))
        print('\nthreshold0=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]' % (threshold0[0], threshold0[1], threshold0[2], threshold0[3], threshold0[4], threshold0[5], threshold0[6], threshold0[7], threshold0[8], threshold0[9], threshold0[10], threshold0[11], threshold0[12], threshold0[13], threshold0[14], threshold0[15], threshold0[16], threshold0[17], threshold0[18], threshold0[19], threshold0[20], threshold0[21], threshold0[22], threshold0[23], threshold0[24], threshold0[25], threshold0[26], threshold0[27], threshold0[28], threshold0[29], threshold0[30], threshold0[31], threshold0[32], threshold0[33], threshold0[34], threshold0[35], threshold0[36], threshold0[37], threshold0[38], threshold0[39]))
        print('threshold1=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]' % (threshold1[0], threshold1[1], threshold1[2], threshold1[3], threshold1[4], threshold1[5], threshold1[6], threshold1[7], threshold1[8], threshold1[9], threshold1[10], threshold1[11], threshold1[12], threshold1[13], threshold1[14], threshold1[15], threshold1[16], threshold1[17], threshold1[18], threshold1[19], threshold1[20], threshold1[21], threshold1[22], threshold1[23], threshold1[24], threshold1[25], threshold1[26], threshold1[27], threshold1[28], threshold1[29], threshold1[30], threshold1[31], threshold1[32], threshold1[33], threshold1[34], threshold1[35], threshold1[36], threshold1[37], threshold1[38], threshold1[39]))
        train_loss, labeled_train_loss, unlabeled_train_loss, use_rate = train(labeled_trainloader, unlabeled_trainloader, model, ema_model, criterion, optimizer, threshold0, threshold1)
        _, _, outputs_new, targets_new = test(labeled_trainloader, model, criterion, mode='train')
        
        threshold0, threshold1 = dynamic_threshold_generate(outputs_new, targets_new, threshold0, threshold1)
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        test_loss, prec1, _, _ = test(test_loader, test_model, criterion, mode='test')

        # Append logger file
        logger.append([lr, train_loss, labeled_train_loss, unlabeled_train_loss, test_loss, use_rate, prec1])

        # TensorboardX 
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'loss x':labeled_train_loss, 'loss u': unlabeled_train_loss, 'test loss': test_loss}, epoch + 1)      
        writer.add_scalars('accuracy', {'Use Rate': use_rate, 'test accuracy': prec1}, epoch + 1)
        
        # Save model
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        if args.use_ema:
            ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            'best_test_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint = args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()
    print('Best test accuracy:', best_prec1)


def train(labeled_trainloader, unlabeled_trainloader, model, ema_model, criterion, optimizer, threshold0, threshold1):
    
    bar = Bar('train', max = len(unlabeled_trainloader))
    batch_time = AverageMeter() 
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()
    select = 0

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for i in range (len(unlabeled_trainloader)):
        try:
            inputs_x, targets_x = labeled_iter.next()
        except:
            labeled_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_iter.next()
        try:
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
        
        # Measure data loading time
        data_time.update(time.time() - end)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking = True)
        inputs_u_w = inputs_u_w.cuda() 
        inputs_u_s = inputs_u_s.cuda() 

        # Sup loss
        outputs_x = model(inputs_x)
        Lx = criterion(outputs_x, targets_x).mean()

        # Generate pseudo-lables of unlabled samples
        outputs_u_w = model(inputs_u_w)       
        targets_u = outputs_u_w.ge(args.boundary_value).float()

        # Select pseudo-lables
        mask = unsup_generate(threshold0, threshold1, outputs_u_w, targets_u)

        # Compute pseudo-lables utilization
        pre_mask = torch.ones(args.train_batch * args.mu,40).cuda()
        select += (mask == pre_mask).sum().item()
        use_rate = (select / ((args.train_data - (args.train_data * args.label_ratio)) * 40)) * 100

        # Unsup loss
        outputs_u_s = model(inputs_u_s)   
        Lu = (criterion(outputs_u_s, targets_u) * mask).mean()

        # Total loss
        loss = Lx + args.lambda_u * Lu

        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # Compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.use_ema:
                ema_model.update(model)
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        # Plot progress
        bar.suffix  ='({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.2f} | Loss_x: {loss:.2f} | Loss_u: {loss:.2f} '.format(
                    batch = i + 1,
                    size = len(unlabeled_trainloader),
                    data = data_time.avg,
                    bt = batch_time.avg,
                    total = bar.elapsed_td,
                    eta = bar.eta_td,
                    loss = losses.avg,
                    loss_x = losses_x.avg,
                    loss_u = losses_u.avg,                        
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, use_rate)


def test(test_loader, model, criterion, mode):

    bar = Bar(f'{mode}', max = len(test_loader))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    correct = 0
    test_acc=[]

    model.eval()

    outputs_new = torch.ones(1, 40).cuda()
    targets_new = torch.ones(1, 40).cuda()

    with torch.no_grad():
        end = time.time()    
        for i, (inputs, targets) in enumerate(test_loader):
            # Measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda(non_blocking = True)
            targets = targets.cuda(non_blocking = True)

            # Compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()

            outputs_new = torch.cat((outputs_new, outputs), dim = 0)
            targets_new = torch.cat((targets_new, targets), dim = 0)

            # Measure accuracy
            results = outputs > 0.5
            if mode == 'train':
                correct += (results == targets).sum().item()
                acc = (correct / (args.train_data * args.label_ratio * 40)) * 100
            else:
                correct += (results == targets).sum().item()
                acc = (correct / (args.test_data * 40)) * 100
            test_acc.append(acc)

            losses.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # Plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | prec1: {prec1: .4f}'.format(
                    batch = i + 1,
                    size = len(test_loader),
                    data = data_time.avg,
                    bt = batch_time.avg,
                    total = bar.elapsed_td,
                    eta = bar.eta_td,
                    loss = losses.avg,
                    prec1 = acc
                    )
            bar.next()
        bar.finish()

    return (losses.avg, acc, outputs_new, targets_new)


# Save the latest and best models
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# Use the predictions of all labeled data to obtain dynamic dual thresholds
def dynamic_threshold_generate(outputs_new, targets_new, threshold0, threshold1):
    outputs_l = outputs_new[1:, :]
    targets_l = targets_new[1:]
    targets_t = ~(targets_l.bool()) + 0
    pred0 = outputs_l * targets_l
    pred1 = outputs_l * targets_t

    for i in range(40):
        outputs0_new = pred0[:, i][torch.nonzero(pred0[:, i])].view(-1) # Presence
        outputs1_new = pred1[:, i][torch.nonzero(pred1[:, i])].view(-1) # Absence  
        threshold0[i] = 0.5 * (outputs0_new.min() + outputs0_new.max()) 
        threshold1[i] = 0.5 * (outputs1_new.min() + outputs1_new.max()) 

    return threshold0, threshold1


# Select pseudo-labels
def unsup_generate(threshold0, threshold1, outputs_u_w, targets_u): 
    mask = []
    dim0, dim1 = targets_u.shape
    for m in range(dim0):
        for n in range(dim1):
            if  targets_u[m, n] == 1:   #Pseudo-label indicating the presence of an attribute
                mask.append(outputs_u_w[m, n].ge(threshold0[n]).float())
            else:    
                mask.append(outputs_u_w[m, n].lt(threshold1[n]).float())
    mask=torch.as_tensor(mask).reshape(dim0, dim1)
    return mask.cuda()


# Adjust learning rate
def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** (epoch // args.step))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * epoch / args.epochs)) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - epoch / args.epochs)
    elif args.lr_decay == 'linear2exp':
        if epoch < args.turning_point + 1:
            # learning rate decay as 95% at the turning point (1 / 95% = 1.0526)
            lr = args.lr * (1 - epoch / int(args.turning_point * 1.0526))
        else:
            lr *= args.gamma
    elif args.lr_decay == 'schedule':
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
