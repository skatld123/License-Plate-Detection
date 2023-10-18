_base_ = '../dino/dino-5scale_swin-l_8xb2-12e_coco.py'
max_epochs = 50
dataset_type = 'CLP'
data_root = '/root/dataset_clp/dataset_2044_new/'
work_dir = '/root/mmdetection/work_dirs/dino_2044_new_' + str(max_epochs)
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
        interval=5,
        save_best='auto',
        out_dir=work_dir)
)

test_evaluator = dict(
    outfile_prefix='./work_dirs/clp_detection/dino_2044_new_50/')