# 훈련 
# bash ./tools/dist_train.sh \
#     /root/mmdetection/configs/clp/swin_clp.py \
#     2 \
# 테스트
# ./tools/dist_test.sh \
#     configs/clp/dcnv2_clp.py \
#     /mmdetection/work_dirs/dcnv2_2044_50/best_coco_bbox_mAP_epoch_19.pth \
#     2 \
#     --out result_pkl/dcn_2044_50.pkl
# 결과 분석 (로그)
# python tools/analysis_tools/analyze_results.py \
#        /root/mmdetection/configs/clp/swin_clp.py \
#        /root/mmdetection/result/swin_2044_200.pkl \
#        /root/mmdetection/results/swin/ \
#        --topk 150 --show-score-thr 0.5 
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn_clp.py',
    '../_base_/datasets/clp_detection.py',
    '../_base_/schedules/schedule_clp.py', '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

max_epochs = 50

work_dir = '/root/mmdetection/work_dirs/swin_2044_' + str(max_epochs)


model = dict(
    # type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

# augmentation strategy originates from DETR / Sparse RCNN
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     # dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[[
#             dict(
#                 type='RandomChoiceResize',
#                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                         (736, 1333), (768, 1333), (800, 1333)],
#                 keep_ratio=True)
#         ],
#                     [
#                         dict(
#                             type='RandomChoiceResize',
#                             scales=[(400, 1333), (500, 1333), (600, 1333)],
#                             keep_ratio=True),
#                         dict(
#                             type='RandomCrop',
#                             crop_type='absolute_range',
#                             crop_size=(384, 600),
#                             allow_negative_crop=True),
#                         dict(
#                             type='RandomChoiceResize',
#                             scales=[(480, 1333), (512, 1333), (544, 1333),
#                                     (576, 1333), (608, 1333), (640, 1333),
#                                     (672, 1333), (704, 1333), (736, 1333),
#                                     (768, 1333), (800, 1333)],
#                             keep_ratio=True)
#                     ]]),
#     dict(type='PackDetInputs')
# ]
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggetHook'),
        dict(type='TensorboardLoggerHook')
    ])
log_level = 'INFO'
dist_params = dict(backend='nccl')


default_hooks = dict(
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=10,
        min_delta=0.005),
    checkpoint=dict(
        type="CheckpointHook",
        save_best='auto',
        out_dir=work_dir)
)