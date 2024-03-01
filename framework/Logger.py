import os
from datetime import datetime
import json 
import warnings

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
        
        self.exp_name = exp_name
        with open(self.path, 'w') as writer:
            writer.write('EXP NAME:'+ exp_name+ '\n')
            writer.write('CONFIG USED\n')
            writer.write(cfg_str+'\n')
            print('Config\n', cfg_str)
        
        self.initialized = False
        
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
    
    def train_step(self, epoch, max_epoch, batch_index, total_batches, lr, eta, loss_dict, total_loss):
        message = 'Epoch [{epoch}/{max_epoch}][{batch_index}/{total_batches}]'.format(
            epoch=epoch,
            max_epoch=max_epoch,
            batch_index=batch_index,
            total_batches=total_batches
        )
        message += 'lr: {},'.format(lr)
        message += 'eta: {eta},'.format(eta)
        
        for (k,v) in loss_dict:
            message += '{k}: {v},'.format(k=k,v=v)
        
        message += 'loss: {total_loss}\n'.format(
            total_loss=total_loss
            )
    
    
    def print_file(self, message):
        with open(self.path, 'a') as writer:
            writer.write(message+'\n')
            
logger = Logger()