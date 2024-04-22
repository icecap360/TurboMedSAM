from torchvision.transforms.v2 import CutMix
import torch

class NoLabelCutMix(CutMix):
    """Apply CutMix to the provided batch of images and labels.

    Paper: `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    <https://arxiv.org/abs/1905.04899>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``CutMix()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """
    def __init__(self, *, alpha: float = 1.0, labels_getter="default") -> None:
        return super().__init__(alpha=alpha,labels_getter=labels_getter, num_classes=1)
        
    def forward(self, *inputs):
        batch_size = len(inputs[0]['meta']['idx'])
        placeholder_labels = torch.zeros((batch_size,), dtype=torch.int64)
        inputs.append(placeholder_labels)
        super().forward(inputs)
        
    # def _transform(self, inpt, params):
    #     if inpt is params["labels"]:
    #         raise Exception('Not for labels')
    #     super()._transform(inpt, params)

def NoLabelCutMixFunctional(x, params):
    batch_size = len(x)
    labels = torch.zeros((batch_size,), dtype=torch.int64)
    
    params['labels'] = labels
    params["batch_size"] = batch_size
    
    x1, y1, x2, y2 = params["box"]
    rolled = x.roll(1, 0)
    output = x.clone()
    output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]
    return output

if __name__ == '__main__':
    from torchvision.transforms import functional as F
    import matplotlib.pyplot as plt
    torch.random.manual_seed(1)
    im1 = torch.zeros((1,1024,1024), dtype=torch.float32)
    im2 = torch.ones((1,1024,1024), dtype=torch.float32)
    batch = torch.stack((im1, im2))
    mixpatch = NoLabelCutMix( alpha=1.0)
    params = mixpatch._get_params(batch)
    print(params)
    results = NoLabelCutMixFunctional(batch, params)*255
    results = results.type(torch.uint8)
    disp = F.to_pil_image(results[1])
    print(disp)
    disp.save('trash.png')