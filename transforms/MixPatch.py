import torch
import numpy as np
import math
from torchvision.transforms.v2.functional import get_dimensions, get_size, is_pure_tensor
from torchvision import tv_tensors
import PIL.Image

class MixPatch(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, alpha, patch_size):
        self.alpha = alpha
        self.patch_size = patch_size

    def mixup_patch(self, x, alpha, patch_size, use_cuda=True,):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        '''Copied from https://github.com/Xiang-Deng-DL/PEBKD/blob/master/helper/loops.py#L62'''
        #alpha = opt.mix_alpha
        #path_size = opt.patch_size
        
        batch_size, channel, h, w = x.shape
        patch_num = int(h/patch_size)*int(w/patch_size)
    
        #lam = np.random.beta(alpha, alpha, size=(batch_size, path_num) )##64, 16, 1, 1, 1
        
        lam = np.random.beta(alpha, alpha, size=(1, patch_num) )

        
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
            lam= torch.tensor(lam).cuda()
            lam = lam.float()
        else:
            index = torch.randperm(batch_size)
            lam= torch.tensor(lam)
            lam = lam.float()
        
        x = x.view(batch_size, channel, int(h/patch_size), patch_size, int(w/patch_size), patch_size)#64, 3, 4, 8, 4, 8
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()#64, 3, 4, 4, 8, 8
        x = x.view(batch_size, channel, patch_num, patch_size, patch_size)# 64, 3, 16, 8, 8
        x = x.permute(0, 2, 1, 3, 4).contiguous()# 64,16, 3, 8, 8
        #lam = lam.reshape(batch_size, path_num, 1, 1, 1)
        lam = lam.reshape(1, patch_num, 1, 1, 1)
            
        mixed_x = lam * x + (1.0 - lam) * x[index, :]
        mixed_x = mixed_x.permute(0, 2, 1, 3, 4).contiguous() # 64,3, 16, 8, 8
        mixed_x = mixed_x.view(batch_size, channel, int(h/patch_size), int(w/patch_size), patch_size, patch_size) # 64, 3, 4, 4, 8, 8
        mixed_x = mixed_x.permute(0, 1, 2, 4, 3, 5).contiguous() # 64, 3, 4, 8, 4, 8
        mixed_x = mixed_x.view(batch_size, channel, h, w)
        
        return mixed_x

    def __call__(self, batch):
        images = batch['image']
        batch = self.mixup_patch(images, self.alpha, self.patch_size, True)
        return batch
    
    def _get_params(self, flat_inputs):
        batch_size, channel, h, w = flat_inputs.shape
        patch_num = int(h/self.patch_size)*int(w/self.patch_size)
    
        #lam = np.random.beta(alpha, alpha, size=(batch_size, path_num) )##64, 16, 1, 1, 1
        
        lam = np.random.beta(self.alpha, self.alpha, size=(1, patch_num) )

        index = torch.randperm(batch_size)
        lam= torch.tensor(lam)
        lam = lam.float()
        
        #lam = lam.reshape(batch_size, path_num, 1, 1, 1)
        lam = lam.reshape(1, patch_num, 1, 1, 1)
        
        return dict(lam=lam, index=index, patch_size=self.patch_size)

    def query_size(self, flat_inputs):
        sizes = {
            tuple(get_size(inpt))
            for inpt in flat_inputs
            if self.check_type(
                inpt,
                (
                    is_pure_tensor,
                    tv_tensors.Image,
                    PIL.Image.Image,
                    tv_tensors.Video,
                    tv_tensors.Mask,
                    tv_tensors.BoundingBoxes,
                ),
            )
        }
        if not sizes:
            raise TypeError("No image, video, mask or bounding box was found in the sample")
        elif len(sizes) > 1:
            raise ValueError(f"Found multiple HxW dimensions in the sample: {self.sequence_to_str(sorted(sizes))}")
        h, w = sizes.pop()
        return h, w
    
    def check_type(self, obj, types_or_checks) -> bool:
        for type_or_check in types_or_checks:
            if isinstance(obj, type_or_check) if isinstance(type_or_check, type) else type_or_check(obj):
                return True
        return False
    
    def sequence_to_str(self, seq, separate_last: str = "") -> str:
        if not seq:
            return ""
        if len(seq) == 1:
            return f"'{seq[0]}'"

        head = "'" + "', '".join([str(item) for item in seq[:-1]]) + "'"
        tail = f"{'' if separate_last and len(seq) == 2 else ','} {separate_last}'{seq[-1]}'"

        return head + tail

def MixPatchFunctional(x, params, use_cuda=True):
    patch_size = params['patch_size']
    lam = params['lam']
    index = params['index']
    
    batch_size, channel, h, w = x.shape
    patch_num = int(h/patch_size)*int(w/patch_size)
    
    x = x.view(batch_size, channel, int(h/patch_size), patch_size, int(w/patch_size), patch_size) #64, 3, 4, 8, 4, 8
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous() #64, 3, 4, 4, 8, 8
    x = x.view(batch_size, channel, patch_num, patch_size, patch_size) # 64, 3, 16, 8, 8
    x = x.permute(0, 2, 1, 3, 4).contiguous() # 64,16, 3, 8, 8

    lam = lam.reshape(1, patch_num, 1, 1, 1)
    
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    mixed_x = mixed_x.permute(0, 2, 1, 3, 4).contiguous() # 64,3, 16, 8, 8
    mixed_x = mixed_x.view(batch_size, channel, int(h/patch_size), int(w/patch_size), patch_size, patch_size) # 64, 3, 4, 4, 8, 8
    mixed_x = mixed_x.permute(0, 1, 2, 4, 3, 5).contiguous() # 64, 3, 4, 8, 4, 8
    mixed_x = mixed_x.view(batch_size, channel, h, w)
    
    return mixed_x

if __name__ == '__main__':
    from torchvision.transforms import functional as F
    import matplotlib.pyplot as plt
    torch.random.manual_seed(1)
    im1 = torch.zeros((1,1024,1024), dtype=torch.float32)
    im2 = torch.ones((1,1024,1024), dtype=torch.float32)
    batch = torch.stack((im1, im2))
    mixpatch = MixPatch( alpha=1.0, patch_size=256)
    params = mixpatch._get_params(batch)
    print(params)
    results = MixPatchFunctional(batch, params)*255
    results = results.type(torch.uint8)
    disp = F.to_pil_image(results[1])
    print(disp)
    disp.save('trash.png')
    # plt.imshow(  disp )
    # plt.show()