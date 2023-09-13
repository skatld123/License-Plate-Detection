# bash ./tools/dist_train.sh \
#     /root/mmdetection/configs/car/dcn_faster-rcnn_r101-dconv-c3-c5_fpn_1x_car.py \
#     2 \
_base_ = '../faster_rcnn/faster-rcnn_r101_fpn_1x_car.py'

dataset_type = 'Car'
data_root = '/root/dataset_clp/dataset_syncar_coco/'

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggetHook'),
        # dict(type='TensorboardLoggerHook')
    ])
log_level = 'INFO'
dist_params = dict(backend='nccl')