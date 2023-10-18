_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'
max_epochs = 50
dataset_type = 'CLP'
data_root = '/root/dataset_clp/dataset_2044_new/'
work_dir = '/root/mmdetection/work_dirs/cascade_rcnn_2044_new_' + str(max_epochs)

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

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

# sh
# 훈련
# bash ./tools/dist_train.sh     /root/mmdetection/configs/clp/cascade-rcnn_x101-64x4d_fpn_1x_coco.py     2
# 테스트
# ./tools/dist_test.sh \
#     configs/clp/cascade-rcnn_x101-64x4d_fpn_1x_coco..py \
#         work_dirs/cascade_rcnn_2044_new_50/best_coco_bbox_mAP_epoch_10.pth \
#         2 \
#          --out result_pkl/casc_2044_10.pkl
# 시각화
# python tools/visualizations/vis_cam.py \
#     demo/bird.JPEG \
#     configs/resnet/resnet50_8xb32_in1k.py \
#     https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth \
#     --method GradCAM