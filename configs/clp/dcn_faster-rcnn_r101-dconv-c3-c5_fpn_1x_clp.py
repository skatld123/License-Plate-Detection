_base_ = '../faster_rcnn/faster-rcnn_r101_fpn_1x_clp.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
