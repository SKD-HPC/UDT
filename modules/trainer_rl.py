import copy
import logging
import os
import time
from abc import abstractmethod
import pandas as pd
import torch
from numpy import inf
from modules.optimizers import set_lr
from modules.rewards import get_self_critical_reward, init_scorer
from modules.loss import compute_loss
import csv
import os

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args):
        self.args = args            
        self.logger = logging.getLogger(__name__)
    
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.ve_optimizer = ve_optimizer
        self.ed_optimizer = ed_optimizer
        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_best(self, epoch, log):
     
        best = False
        if self.mnt_mode != 'off':
            try:
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                           (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                self.logger.warning(
                    "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                self.mnt_mode = 'off'
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric]
                best = True

            self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)
  
    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _get_learning_rate(self):
        lrs = list()
        lrs.append(self.ve_optimizer.current_lr)
        lrs.append(self.ed_optimizer.current_lr)
        return {'lr_visual_extractor': lrs[0], 'lr_encoder_decoder': lrs[1]}

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            've_optimizer': self.ve_optimizer.state_dict(),
            'ed_optimizer': self.ed_optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, ve_optimizer, ed_optimizer, args)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _set_lr_ve(self, iteration):
        if iteration < self.args.noamopt_warmup:
            current_lr = self.args.lr_ve * (iteration + 1) / self.args.noamopt_warmup
            set_lr(self.ve_optimizer, current_lr)

    def _set_lr_ed(self, iteration):
        if iteration < self.args.noamopt_warmup:
            current_lr = self.args.lr_ed * (iteration + 1) / self.args.noamopt_warmup
            set_lr(self.ed_optimizer, current_lr)

    def _train_epoch(self, epoch):
        
        train_loss = 0
        self.model.train()
        tag_ids = self.model.tokenizer.get_tag_id().to(self.device)
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
            init_scorer()
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(images, tag_ids, mode='sample',
                                           update_opts={'sample_method': self.args.sc_sample_method,
                                                        'beam_size': self.args.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(images, tag_ids, mode='sample',
                                                     update_opts={'sample_method': self.args.train_sample_method,
                                                                  'beam_size': self.args.train_beam_size,
                                                                  'sample_n': self.args.train_sample_n})

            gts = reports_ids[:, 1:]
            reward = get_self_critical_reward(greedy_res, gts, gen_result)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            '''
            #########################################################
            # Meric-based Focal Learning
            alpha = torch.tanh(reward)
            f = (1-alpha) ** 2 + 1 
            reward = f * reward
            #########################################################
            '''
            loss_rl = self.criterion(sample_logprobs, gen_result.data, reward)

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
            output = self.model(images, tag_ids, reports_ids, mode='train')
            loss_nll = compute_loss(output, reports_ids, reports_masks)
            loss = 0.01 * loss_nll + 0.99 * loss_rl
            train_loss += loss.item()
            self.ve_optimizer.zero_grad()
            self.ed_optimizer.zero_grad()
            loss.backward()
            self.ve_optimizer.step()
            self.ed_optimizer.step()
        
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        log.update({'seed':self.args.seed})
        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, tag_ids, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []

            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, tag_ids, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
       
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        
        return log
