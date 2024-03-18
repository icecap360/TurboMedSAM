import os
from datetime import datetime
import json 
import warnings
import torch 
import time

class Logger:
    def __init__(self):
        self.start_time = time.time()

    def init_message(self, work_dir, exp_name, cfg_str):
        
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d%m%Y-%H%M")
        self.path = os.path.join(
            work_dir,
            formatted_datetime+'.log'
        )
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.exp_name = exp_name
        with open(self.path, 'w') as writer:
            writer.write('EXP NAME:'+ exp_name+ '\n')
            writer.write('CONFIG USED\n')
            writer.writelines(cfg_str)
            for line in cfg_str:
                print(line, end='')
        print('\n')
        self.start_time = time.time()
        
    def log(self, message, print_to_file, level='INFO'):
        if type(message) is dict:
            message = json.dumps(message)
        elif not (type(message) is str):
            message = str(message)
        
        if level != 'INFO':
            message = level + ': ' + message

        if level == 'WARN':
            warnings.warn(message)
        print( message)
        if print_to_file:
            self.print_file( message)
    
    def info(self, message):
        self.log(message, print_to_file=False, level='INFO')
    
    def info_and_print(self, message):
        self.log(message, print_to_file=True, level='INFO')

    def print_screen(self, message):
        print(message)
    
    def calc_eta_epoch(self, epoch, max_epoch, batch_index, total_batches):
        time_now = time.time()
        percent_done = epoch/max_epoch + batch_index/(total_batches*max_epoch)
        elapsed = time_now - self.start_time
        total_time = elapsed / percent_done
        rem_time_sec = total_time - elapsed
        hours = int(rem_time_sec // 3600)
        minutes = int((rem_time_sec % 3600) // 60)
        seconds = int(rem_time_sec % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def calc_eta_iter(self, iters, max_iters):
        time_now = time.time()
        percent_done = iters/max_iters 
        elapsed = time_now - self.start_time
        total_time = elapsed / percent_done
        rem_time_sec = total_time - elapsed
        hours = int(rem_time_sec // 3600)
        minutes = int((rem_time_sec % 3600) // 60)
        seconds = int(rem_time_sec % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def train_epoch_message(self, epoch, rank, max_epoch, samples_processed, total_samples, lr, loss_dict, total_loss):
        message = 'Epoch rank-{rank} [{epoch}/{max_epoch}][{batch_index}/{total_batches}] '.format(
            epoch=epoch,
            max_epoch=max_epoch,
            batch_index=samples_processed,
            total_batches=total_samples,
            rank=rank,
        )
        message += '    lr: {}, '.format(lr)
        
        eta = self.calc_eta_epoch(epoch, max_epoch, samples_processed, total_samples)
        message += 'eta: ' + eta + ', '
        
        loss_dict_message = ''
        for (k,v) in loss_dict.items():
            loss_dict_message += '{k}: {v:05f}, '.format(k=k,v=v)
        
        if type(total_loss) is torch.Tensor:
            total_loss = total_loss.item()
        message += loss_dict_message
        message += '   -   total loss: {total_loss}'.format(
            total_loss=total_loss
            )
        return message
    
    def train_iter_message(self, rank, iters, max_iters, lr, loss_dict, total_loss):
        message = 'Epoch rank-{rank} [{iters}/{max_iters}][{percent:02f}] '.format(
            iters=iters,
            max_iters=max_iters,
            percent=iters/max_iters,
            rank=rank,
        )
        message += '    lr: {}, '.format(lr)
        
        eta = self.calc_eta_iter(iters, max_iters)
        message += 'eta: ' + eta + ', '
        
        loss_dict_message = ''
        for (k,v) in loss_dict.items():
            loss_dict_message += '{k}: {v}, '.format(k=k,v=v)
        
        if type(total_loss) is torch.Tensor:
            total_loss = total_loss.item()
        message += loss_dict_message
        message += '   -   total loss: {total_loss}'.format(
            total_loss=total_loss
            )
        return message + '\n'

    def val_epoch_message(self, loader_name, epoch, max_epoch, loss_dict, total_loss, metrics_dict):
        message = '\n VALIDATION {loader_name}:   Epoch [{epoch}/{max_epoch}]\n'.format(
            loader_name=loader_name,
            epoch=epoch,
            max_epoch=max_epoch,
        )
        
        message += 'loss_dict - '

        for (k,v) in loss_dict.items():
            if type(v) is torch.Tensor:
                v = v.item()
            message += '{k}: {v},'.format(k=k,v=v)
        
        if type(total_loss) is torch.Tensor:
            total_loss = total_loss.item()
        message += '\ntotal loss: {total_loss}\n'.format(
            total_loss=total_loss
            )
        message += 'metrics - '
        for (k,v) in metrics_dict.items():
            if type(v)is torch.Tensor:
                v = v.item()
            message += '{k}: {v},'.format(k=k,v=v)
        return message + '\n'

    def val_iter_message(self, loader_name, iters, max_iters, loss_dict, total_loss, metrics_dict):
        message = '\nVALIDATION {loader_name}: [{iters}/{max_iters}][{percent}] '.format(
            loader_name=loader_name,
            iters=iters,
            max_iters=max_iters,
            percent=iters/max_iters,
        )
        message += 'loss_dict - '

        for (k,v) in loss_dict.items():
            if type(v) is torch.Tensor:
                v = v.item()
            message += '{k}: {v}, '.format(k=k,v=v)
        
        if type(total_loss) is torch.Tensor:
            total_loss = total_loss.item()
        message += '\ntotal loss: {total_loss}\n'.format(
            total_loss=total_loss
            )
        message += 'metrics - '
        for (k,v) in metrics_dict.items():
            if type(v)is torch.Tensor:
                v = v.item()
            message += '{k}: {v},'.format(k=k,v=v)
        return message + '\n'
        
    def print_file(self, message):
        with open(self.path, 'a') as writer:
            writer.write(message+'\n')
            
logger = Logger()