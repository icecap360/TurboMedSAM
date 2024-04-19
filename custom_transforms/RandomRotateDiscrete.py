import torch
import random
import torchvision.transforms.functional as TF
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform
from typing import Any, Dict

class RandomRotateDiscrete(Transform):
    
    def __init__(self, *, angles=[-90,90,0,180]) -> None:
        super().__init__()
        self.angles = angles

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.rotate,
            inpt,
            angle=params['angle']
        )
    
    # def __call__(self, inputs, targets):
    #     images = inputs['image']
    #     boxes = inputs['bbox']
    #     masks = targets['masks']
    #     angle = random.choice(self.angles)
    #     inputs['image'] = TF.rotate(images, angle)
    #     return inputs, targets
    def _get_params(self, flat_inputs):
        angle = random.choice(self.angles)
        return dict(angle=angle)
        
    # def _transform(self, inpt, params):
    #     if inpt is params["labels"]:
    #         raise Exception('Not for labels')
    #     super()._transform(inpt, params)

def RandomRotateDiscreteFunctional(x, params):
    return TF.rotate(x, params['angle'])

if __name__ == '__main__':
    from torchvision.transforms import functional as F
    import matplotlib.pyplot as plt
    torch.random.manual_seed(1)
    from PIL import Image

    image = Image.open("/home/qasim/Projects/TurboMedSAM/plots/im1.jpeg")
    # Resize the image to 1024x1024
    resized_image = TF.resize(image, (1024, 1024))
    # Convert the resized image to a NumPy array
    numpy_array = TF.to_tensor(resized_image)
    # Convert the NumPy array to a PyTorch tensor
    torch_tensor = numpy_array.type(torch.float32)
    
    im1 = torch_tensor
    im2 = torch.ones((3,1024,1024), dtype=torch.float32)
    batch = torch.stack((im1, im2))
    rotater = RandomRotateDiscrete()
    params = rotater._get_params(batch)
    print(params)
    params = {'angle':90}
    results = RandomRotateDiscreteFunctional(batch, params)*255
    results = results.type(torch.uint8)
    disp = F.to_pil_image(results[0])
    print(disp)
    disp.save('/home/qasim/Projects/TurboMedSAM/plots/trash.png')