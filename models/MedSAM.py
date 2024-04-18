
import torch
import torch.nn as nn
from .projects.LiteMedSAM.segment_anything.modeling import Sam
from framework import BaseModule, keep_keys

class MedSAM(BaseModule):
    def __init__(
        self,
        image_size,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        init_cfg=None
    ) -> None:
        super().__init__( init_cfg)
        self.image_size=image_size
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        
        
    @torch.no_grad()
    def forward(
        self,
        batched_input,
        multimask_output: bool =False,
    ):
        input_images = batched_input['image']
        # torch.stack(
        #     [self.preprocess(x["image"]) for x in batched_input], dim=0
        # )
        image_embeddings = self.image_encoder(input_images)

        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=batched_input['bbox'],
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # masks = self.postprocess_masks(
        #     low_res_masks,
        #     input_size=(1024,1024),
        #     original_size=(512,512),
        # )
        # masks = masks > self.mask_threshold
        # outputs['masks'].append(masks)
        
        return {
            "logits": low_res_masks, 
            "iou": iou_predictions
        }
    def init_weights(self, state_dict=None, strict=True):
        if self.init_cfg == None and state_dict == None:
            return
        elif state_dict:
            self.prompt_encoder.load_state_dict(
                keep_keys(state_dict, "prompt_encoder", [(r"^prompt_encoder.", "")]),
                strict= strict
            )
            self.mask_decoder.load_state_dict(
                keep_keys(state_dict, "mask_decoder", [(r"^mask_decoder.", "")]),
                strict=strict
            )
            self.image_encoder.load_state_dict(
                keep_keys(state_dict, "image_encoder", [(r"^image_encoder.", "")]),
                strict= strict
            )
        elif 'pretrained' in self.init_cfg['type'].lower():
            if not 'checkpoint' in self.init_cfg.keys():
                raise Exception('Missing checkpoint')  
            state_dict = torch.load(self.init_cfg['checkpoint'])
            self.prompt_encoder.load_state_dict(
                keep_keys(state_dict, "prompt_encoder", [(r"^prompt_encoder.", "")]),
                strict=self.init_cfg.get('strict') or True
            )
            self.mask_decoder.load_state_dict(
                keep_keys(state_dict, "mask_decoder", [(r"^mask_decoder.", "")]),
                strict=self.init_cfg.get('strict') or True
            )
            if not self.init_cfg.get('no_image_encoder'):
                self.image_encoder.load_state_dict(
                    keep_keys(state_dict, "image_encoder", [(r"^image_encoder.", "")]),
                    strict=self.init_cfg.get('strict') or True
                )
            else:
                self.image_encoder.init_weights()
        else:
            raise Exception('init_cfg is formatted incorrectly')

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size,
        original_size
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x