# bash ./tools/dist_train.sh     /root/mmdetection/configs/clp/dcnv2_clp.py     2
_base_ = '../dino/dino-5scale_swin-l_8xb2-12e_coco.py'
max_epochs = 150
dataset_type = 'CLP'
data_root = '/root/dataset_clp/dataset_crop/'
work_dir = '/root/mmdetection/work_dirs/dino_crop_aug_' + str(max_epochs)
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

geometric_space = [
    dict(type='ShearX', prob=0.2),
    dict(type='ShearY', prob=0.2),
    dict(type='Rotate', prob=0.2)
]

color_space = [
    dict(type='Contrast', prob=0.2),
    dict(type='Brightness', prob=0.2),
    dict(type='Sharpness', prob=0.2),
    dict(type='Equalize', prob=0.2)
]

# RANDAUG_SPACE = [[dict(type='AutoContrast')], [dict(type='Equalize')],
#                  [dict(type='Invert')], [dict(type='Rotate')],
#                  [dict(type='Posterize')], [dict(type='Solarize')],
#                  [dict(type='SolarizeAdd')], [dict(type='Color')],
#                  [dict(type='Contrast')], [dict(type='Brightness')],
#                  [dict(type='Sharpness')], [dict(type='ShearX')],
#                  [dict(type='ShearY')], [dict(type='TranslateX')],
#                  [dict(type='TranslateY')]]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_num=1, prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    # crop_size=(384, 600),
                    crop_size=(600, 600),
                    allow_negative_crop=True),
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(1333, 608), (1333, 640), (1333, 672), (1333, 704),
                            (1333, 736), (1333, 768), (1333, 800),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    # crop_size=(384, 600),
                    crop_size=(600, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
            ]
        ]),
    dict(type='PackDetInputs')
]

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
        patience=15,
        min_delta=0.005),
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        save_best='auto',
        out_dir=work_dir)
)

test_evaluator = dict(
    outfile_prefix='./work_dirs/clp_detection/dino_crop_aug/')