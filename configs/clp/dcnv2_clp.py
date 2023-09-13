# Deformable Conv Networks 2
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_clp.py'
work_dir = '/root/mmdetection/work_dirs/dcnv2_2044_200'
max_epoch = 200
model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                _delete_=True,
                type='ModulatedDeformRoIPoolPack',
                output_size=7,
                output_channels=256),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32])))

# python tools/analysis_tools/analyze_results.py \
#        /root/mmdetection/configs/clp/dcnv2_clp.py \
#        /root/mmdetection/result/dcnv2_2044_100.pkl \
#        /root/mmdetection/results/dcnv2/ \

# bash ./tools/dist_train.sh     /root/mmdetection/configs/clp/dcnv2_clp.py     2

# ./tools/dist_test.sh \
#     configs/clp/dcnv2_clp.py \
#         work_dirs/dcnv2_OPEN_pretrained_epoch100/epoch_100.pth \
#         2 \
#          --out work_dirs/result/dcnv2_detection/results_open.pkl

#  python tools/analysis_tools/analyze_logs.py  plot_curve  work_dirs/dcnv2_OPEN_pretrained_epoch100/20230606_064555/vis_data/20230606_064555.json  \
#     --keys bbox_mAP \
#     --legend dcnv2_open_pt_100 \
#     --out work_dirs/result/dcnv2_detection/dcnv2_open_mAP.png

# python tools/analysis_tools/analyze_results.py \
#     configs/clp/dcnv2_clp.py \
#     work_dirs/result/dcnv2_detection/results_open.pkl \
#     work_dirs/result/dcnv2_detection/images_open \
#     --topk 50
