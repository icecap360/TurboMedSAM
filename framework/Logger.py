import os
from datetime import datetime
import json 
import warnings
import torch 
import time

class Logger:

    def init(self, work_dir, exp_name, cfg_str, print_screen_off=False):
        self.print_screen_off = print_screen_off
        
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d%m%Y-%H%M")
        self.path = os.path.join(
            work_dir,
            exp_name,
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
        
        self.start_time = time.time()
        self.initialized = True
        
    def log(self, message, level='INFO'):
        if not self.initialized:
            raise Exception('Logger: not initialized yet')
        if type(message) is dict:
            message = json.dumps(message)
        elif not (type(message) is str):
            message = str(message)
        
        if level != 'INFO':
            message = level + ': ' + message
        
        if level == 'WARN':
            warnings.warn(message)
        elif not self.print_screen_off:
            print( message)
        self.print_file( message)
    
    def info(self, message):
        self.log(message,  level='INFO')
    
    def print_screen(self, message):
        print(message)
    
    def calc_eta(self, epoch, max_epoch, batch_index, total_batches):
        time_now = time.time()
        percent_done = epoch/max_epoch + batch_index/(total_batches*max_epoch)
        elapsed = time_now - self.start_time
        total_time = elapsed / percent_done
        rem_time_sec = total_time - elapsed
        hours = int(rem_time_sec // 3600)
        minutes = int((rem_time_sec % 3600) // 60)
        seconds = int(rem_time_sec % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def train_message(self, epoch, rank, max_epoch, samples_processed, total_samples, lr, loss_dict, total_loss):
        message = 'Epoch [{epoch}/{max_epoch}][{batch_index}/{total_batches}] rank{rank} '.format(
            epoch=epoch,
            max_epoch=max_epoch,
            batch_index=samples_processed,
            total_batches=total_samples,
            rank=rank,
        )
        message += '    lr: {}, '.format(lr)
        
        eta = self.calc_eta(epoch, max_epoch, samples_processed, total_samples)
        message += 'eta: ' + eta + ', '
        
        loss_dict_message = ''
        for (k,v) in loss_dict.items():
            loss_dict_message += '{k}: {v}, '.format(k=k,v=v)
        
        if type(total_loss) is torch.Tensor:
            total_loss = total_loss.item()
        message += 'loss: {total_loss}'.format(
            total_loss=total_loss
            )
        message += '\n' + loss_dict_message
        return message
    
    def val_message(self, epoch, max_epoch, loss_dict, total_loss, metrics_dict):
        message = '\n VALIDATION \n\n'
        message = '\n VALIDATION:   Epoch [{epoch}/{max_epoch}]\n'.format(
            epoch=epoch,
            max_epoch=max_epoch,
        )
        
        for (k,v) in loss_dict.items():
            if type(v) is torch.Tensor:
                v = v.item()
            message += '{k}: {v},'.format(k=k,v=v)
        
        if type(total_loss) is torch.Tensor:
            total_loss = total_loss.item()
        message += 'loss: {total_loss}\n'.format(
            total_loss=total_loss
            )
        
        for (k,v) in metrics_dict.items():
            if type(v)is torch.Tensor:
                v = v.item()
            message += '{k}: {v},'.format(k=k,v=v)
        return message

    def print_file(self, message):
        with open(self.path, 'a') as writer:
            writer.write(message+'\n')
            
logger = Logger()