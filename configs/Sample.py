import torch

model = dict(
    type='LiteMedSAM',
    image_encoder=dict(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64, ## (64, 256, 256)
            128, ## (128, 128, 128)
            160, ## (160, 64, 64)
            320 ## (320, 64, 64) 
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
        ),
    prompt_encoder=dict(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
        ),
    mask_decoder=dict(
        num_multimask_outputs=3,
        transformer_depth=2,
        transformer_embedding_dim=256,
        transformer_mlp_dim=2048,
        transformer_num_heads=8,
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        )
    )
gpus = [0]
dataset = 
eval_metrics = []
loss = 
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.9,
    patience=5,
    cooldown=0
)
checkpoint = ''
resume = False
custom_hooks = []
