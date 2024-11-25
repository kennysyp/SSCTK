import contextlib
import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.Ukan import UKAN
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from vat import VATLoss
from utils.losses import ContrastiveLoss

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/SSCTK', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.02,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--start_confidence', type=float,  default=0.5, help='start_confidence')
parser.add_argument('--end_confidence', type=float,  default=0.75, help='end_confidence')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        #2->5% 4->10% 7->20% 18->50%
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "18":478,"35":940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def generate_confidence_mask_for_multiple_models(softmax_outputs_list, threshold, num_classes):  
    # Input shape: [16,4,224,224]  
    # Softmax outputs shape: [16,0,1,2,3,224,224]  
    mask = torch.zeros_like(softmax_outputs_list[0]).sum(dim=1, keepdim=True)  
    # Process each class  
    for class_idx in range(num_classes):  
        # Calculate confidence scores for each model for the current class  
        confidences = [outputs[:, class_idx].unsqueeze(1) for outputs in softmax_outputs_list]  

        # Calculate which models have confidence scores above the threshold  
        high_confidences = [conf > threshold for conf in confidences]  
        # Calculate which models predict this class  
        predictions = [torch.argmax(outputs, dim=1, keepdim=True) == class_idx for outputs in softmax_outputs_list]  
        
        # If two or more models predict this class AND at least one model has confidence above threshold,  
        # set the pixel to 1 in the mask  
        mask += ((sum(predictions) >= len(softmax_outputs_list)//2 + 1) & (sum(high_confidences) >= 1)).float()  

    # Binarize the mask: set pixels > 0 to 1, otherwise 0  
    mask = (mask > 0).float()  

    return mask  

def apply_mask(image, mask, isconfidence=True):  
    # Convert mask to boolean type  
    mask = mask.bool()  
    if not isconfidence:  
        mask = ~mask  
    # Create a black image with same shape as input  
    black_image = torch.zeros_like(image)  
    # Blend input image with black image using the mask  
    output = torch.where(mask, image, black_image)  
    return output




def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    start_confidence = args.start_confidence
    end_confidece = args.end_confidence

    def create_model(ema=False):
        # Network definition unet
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    def create_vit(ema=False):
        # Network definition transformer
        model = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
        model.load_from(config)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    def create_kan(ema=False):
        # Network definition UKan
        model = UKAN(num_classes=args.num_classes,input_channels=1,deep_supervision=False,
                  img_size=args.patch_size[0], embed_dims=[128, 160, 256],no_kan=False).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    ema_model1 = create_model(ema=True)
    model2 = create_vit()
    ema_model2 = create_vit(ema=True)
    model3 = create_kan()
    ema_model3 = create_kan(ema=True)



    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()
    model3.train()


    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    
                           

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.001)
    optimizer3 = optim.SGD(model3.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    con_loss = ContrastiveLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    best_performance3 = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            

            outputs1= model1(volume_batch)
            outputs1_soft = torch.softmax(outputs1, dim=1)
            
            
            outputs2= model2(volume_batch)
            outputs2_soft = torch.softmax(outputs2, dim=1)
          
            

            outputs3= model3(volume_batch)
            outputs3_soft = torch.softmax(outputs3, dim=1)
           
        
            outputs1_ema = ema_model1(volume_batch)
            outputs1_ema_soft = torch.softmax(outputs1_ema, dim=1)

            outputs2_ema = ema_model2(volume_batch)
            outputs2_ema_soft = torch.softmax(outputs2_ema, dim=1)

            outputs3_ema = ema_model3(volume_batch)
            outputs3_ema_soft = torch.softmax(outputs3_ema, dim=1)


            consistency_weight = get_current_consistency_weight(
                iter_num // 300)

            
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss3 = 0.5 * (ce_loss(outputs3[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            
            
            pseudo_outputs1 = torch.argmax(
                outputs1_soft[args.labeled_bs:].detach(), dim=1, keepdim=True)
            
            pseudo_outputs2 = torch.argmax(
                outputs2_soft[args.labeled_bs:].detach(), dim=1, keepdim=True)
            
            pseudo_outputs3 = torch.argmax(
                outputs3_soft[args.labeled_bs:].detach(), dim=1, keepdim=True)
            
            
            outputs_list = [outputs1_soft,outputs2_soft,outputs3_soft,
                            outputs1_ema_soft,outputs2_ema_soft,outputs3_ema_soft]
            
            threshold = start_confidence + (end_confidece-start_confidence) * (iter_num / max_iterations)
            
            confidence_mask = generate_confidence_mask_for_multiple_models(outputs_list,threshold=threshold,num_classes=num_classes)
           

            loss1_plus = 0.5 * dice_loss(apply_mask(outputs1_soft[:args.labeled_bs],confidence_mask[:args.labeled_bs],isconfidence=False), 
                                         apply_mask(label_batch[:args.labeled_bs].unsqueeze(1),confidence_mask[:args.labeled_bs],isconfidence=False))
            
            loss2_plus = 0.5 * dice_loss(apply_mask(outputs2_soft[:args.labeled_bs],confidence_mask[:args.labeled_bs],isconfidence=False), 
                                         apply_mask(label_batch[:args.labeled_bs].unsqueeze(1),confidence_mask[:args.labeled_bs],isconfidence=False))
            
            loss3_plus = 0.5 * dice_loss(apply_mask(outputs3_soft[:args.labeled_bs],confidence_mask[:args.labeled_bs],isconfidence=False), 
                                         apply_mask(label_batch[:args.labeled_bs].unsqueeze(1),confidence_mask[:args.labeled_bs],isconfidence=False))
        
            
            pseudo_outputs1 = apply_mask(pseudo_outputs1,confidence_mask[args.labeled_bs:])
            pseudo_outputs2 = apply_mask(pseudo_outputs2,confidence_mask[args.labeled_bs:])
            pseudo_outputs3 = apply_mask(pseudo_outputs3,confidence_mask[args.labeled_bs:])
        

        
            outputs1_soft_un = apply_mask(outputs1_soft[args.labeled_bs:],confidence_mask[args.labeled_bs:])
            outputs2_soft_un = apply_mask(outputs2_soft[args.labeled_bs:],confidence_mask[args.labeled_bs:])
            outputs3_soft_un = apply_mask(outputs3_soft[args.labeled_bs:],confidence_mask[args.labeled_bs:])

            #12
            pseudo_supervision12 = dice_loss(
                outputs1_soft_un, pseudo_outputs2)
            #13
            pseudo_supervision13 = dice_loss(
                outputs1_soft_un, pseudo_outputs3)
            #21
            pseudo_supervision21 = dice_loss(
                outputs2_soft_un, pseudo_outputs1)
            
            #31
            pseudo_supervision31 = dice_loss(
                outputs3_soft_un, pseudo_outputs1)
                
            
            con1 = con_loss(outputs1,outputs2)
            con2 = con_loss(outputs1,outputs3)

            

            

            #unet 和ema_unet 计算 MSE
            consistency_loss1 = torch.mean(
                    (outputs1_soft[args.labeled_bs:]-outputs1_ema_soft[args.labeled_bs:])**2)

            consistency_loss2 = torch.mean(
                    (outputs2_soft[args.labeled_bs:]-outputs2_ema_soft[args.labeled_bs:])**2)
            
            consistency_loss3 = torch.mean(
                    (outputs3_soft[args.labeled_bs:]-outputs3_ema_soft[args.labeled_bs:])**2)

            
                    

            model1_loss = loss1 + loss1_plus
            model2_loss = loss2 + loss2_plus
            model3_loss = loss3 + loss3_plus

            pseudo_loss = consistency_weight * (pseudo_supervision12 + pseudo_supervision13  +pseudo_supervision21 +pseudo_supervision31  )

            loss = model1_loss + model2_loss +  model3_loss + pseudo_loss + consistency_loss1 + consistency_loss2 + consistency_loss3 + con1 +con2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            loss.backward()


            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

           


            update_ema_variables(model1,ema_model1,args.ema_decay,iter_num)
            update_ema_variables(model2,ema_model2,args.ema_decay,iter_num)
            update_ema_variables(model3,ema_model3,args.ema_decay,iter_num)


            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer3.param_groups:
                param_group['lr'] = lr_


            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/model3_loss',
                              model3_loss, iter_num)
            writer.add_scalar('loss/pseudo_loss',
                              pseudo_loss,iter_num)
            writer.add_scalar('loss/con1_loss',
                              con1,iter_num)
            logging.info('iteration%d: model1 loss:%f model2 loss:%f model3 loss:%f pseudo_loss:%f pseudo_loss:%f' % (
                iter_num, model1_loss.item(), model2_loss.item(),model3_loss.item(),pseudo_loss.item(),con1.item()))
            
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs2 = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs2[1, ...] * 50, iter_num)
                outputs3 = torch.argmax(torch.softmax(
                    outputs3, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model3_Prediction',
                                 outputs3[1, ...] * 50, iter_num)
                
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                #model1
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_hd951, iter_num)
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)
                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                #model2
                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95',
                                  mean_hd952, iter_num)
                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

                #model3
                model3.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model3, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model3_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model3_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                performance3 = np.mean(metric_list, axis=0)[0]
                mean_hd953 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model3_val_mean_dice',
                                  performance3, iter_num)
                writer.add_scalar('info/model3_val_mean_hd95',
                                  mean_hd953, iter_num)
                if performance3 > best_performance3:
                    best_performance3 = performance3
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model3_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance3, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model3.pth'.format(args.model))
                    torch.save(model3.state_dict(), save_mode_path)
                    torch.save(model3.state_dict(), save_best)

                logging.info(
                    'iteration %d : model3_mean_dice : %f model3_mean_hd95 : %f' % (iter_num, performance3, mean_hd953))
                model3.train()

                

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))
                '''
                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))
                '''
                save_mode_path = os.path.join(
                    snapshot_path, 'model3_iter_' + str(iter_num) + '.pth')
                torch.save(model3.state_dict(), save_mode_path)
                logging.info("save model3 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
