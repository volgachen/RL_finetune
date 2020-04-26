import torch
import sys, os, random, time
import numpy as np
import argparse
import logging
import glob

import utils
from controller import Controller

def parse_args():
    parser = argparse.ArgumentParser("RL-Finetune")
    parser.add_argument('--total_iter', type=int, default=200, help='How many architectures to sample.')
    parser.add_argument('--controller_lr', type=float, default=0.0035)
    parser.add_argument('--num_choices', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--lstm_size', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--controller_tanh_constant', type=float, default=1.10)

    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    args = parser.parse_args()
    
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    return args

def preparelog(args):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def main():
    args = parse_args()
    preparelog(args)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


    controller = Controller(args)
    controller.cuda()

    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        args.controller_lr,
        betas=(0.1,0.999),
        eps=1e-3,
    )

    train_loader, valid_loader, test_loader = get_loaders(args)
    total_loss = utils.AvgrageMeter()
    total_reward = utils.AvgrageMeter()
    total_entropy = utils.AvgrageMeter()

    
    base_model = build_basemodel()
    baseline = model_evaluate(base_model, valid_loader)

    controller.train()
    for step in range(args.total_iter):
        controller_optimizer.zero_grad()
        model_para, log_prob, entropy = controller()

        model = model_transform(base_model, model_para)
        model_finetune(model, train_loader)
        with torch.no_grad():
            reward = model_evaluate(model, valid_loader)


        #if args.entropy_weight is not None:
        #    reward += args.entropy_weight*entropy

        log_prob = torch.sum(log_prob)
        loss = log_prob * (reward - baseline)
        loss = loss.sum()
        loss.backward()
        controller_optimizer.step()

        total_loss.update(loss.item(), 1)
        total_reward.update(reward.item(), 1)
        total_entropy.update(entropy.item(), 1)

        if step % args.report_freq == 0:
            #logging.info('controller %03d %e %f %f', step, loss.item(), reward.item(), baseline.item())
            logging.info('controller %03d %e %f %f', step, total_loss.avg, total_reward.avg, baseline.item())
            #tensorboard.add_scalar('controller/loss', loss, epoch)
            #tensorboard.add_scalar('controller/reward', reward, epoch)
            #tensorboard.add_scalar('controller/entropy', entropy, epoch)

if __name__ == "__main__":
    main()
