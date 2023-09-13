python tools/analysis_tools/analyze_logs.py \
plot_curve \
work_dirs/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/20230521_130401/vis_data/20230521_130401.json \
work_dirs/faster-rcnn_r50_fpn_mdpool_1x_clp/20230512_015125/vis_data/20230512_015125.json \
work_dirs/dcn_faster-rcnn_r101-dconv-c3-c5_fpn_1x_clp/20230522_132907/vis_data/20230522_132907.json \
work_dirs/retinanet_effb3_fpn_8xb4-crop896-1x_clp/20230522_051103/vis_data/20230522_051103.json \
--keys bbox_mAP \
--legend swin dcnv2 dcn efficientNet --out results/result_mAP.png

python tools/analysis_tools/analyze_logs.py \
plot_curve \
work_dirs/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/20230521_130401/vis_data/20230521_130401.json \
work_dirs/faster-rcnn_r50_fpn_mdpool_1x_clp/20230512_015125/vis_data/20230512_015125.json \
work_dirs/dcn_faster-rcnn_r101-dconv-c3-c5_fpn_1x_clp/20230522_132907/vis_data/20230522_132907.json \
work_dirs/retinanet_effb3_fpn_8xb4-crop896-1x_clp/20230522_051103/vis_data/20230522_051103.json \
--keys loss_cls \
--legend swin dcnv2 dcn efficientNet --out results/result_cls.png

python tools/analysis_tools/analyze_logs.py \
plot_curve \
work_dirs/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/20230521_130401/vis_data/20230521_130401.json \
work_dirs/faster-rcnn_r50_fpn_mdpool_1x_clp/20230512_015125/vis_data/20230512_015125.json \
work_dirs/dcn_faster-rcnn_r101-dconv-c3-c5_fpn_1x_clp/20230522_132907/vis_data/20230522_132907.json \
work_dirs/retinanet_effb3_fpn_8xb4-crop896-1x_clp/20230522_051103/vis_data/20230522_051103.json \
--keys loss_bbox \
--legend swin dcnv2 dcn efficientNet --out results/result_bbox.png

python tools/analysis_tools/analyze_logs.py \
plot_curve \
work_dirs/effb3_fpn_8xb4-crop896-1x_clp/train_log/vis_data/20230608_063251.json \
work_dirs/swin_open_100/train_log/vis_data/20230607_064905.json \
--keys loss_bbox \
--legend swin dcnv2 dcn efficientNet --out results/result_bbox.png

python tools/analysis_tools/analyze_logs.py \
plot_curve \
work_dirs/effb3_fpn_8xb4-crop896-1x_clp/train_log/vis_data/20230608_063251.json \
work_dirs/swin_open_100/train_log/vis_data/20230607_064905.json \
--keys loss_cls \
--legend efficientNet swin --out results/result_loss_cls_effi_swin_open.png

python tools/analysis_tools/analyze_logs.py \
plot_curve \
work_dirs/effb3_fpn_8xb4-crop896-1x_clp/train_log/vis_data/20230608_063251.json \
work_dirs/swin_open_100/train_log/vis_data/20230607_064905.json \
--keys bbox_mAP \
--legend efficientNet swin --out results/result_mAP_cls_effi_swin_open.png

python tools/analysis_tools/analyze_logs.py \
plot_curve \
/root/mmdetection/work_dirs/dcnv2_2044_200/20230817_013413/vis_data/20230817_013413.json \
--keys bbox_mAP \
--legend dcnv2 --out results/result_mAP.png


# 학습 결과 비교
python tools/analysis_tools/analyze_logs.py \
plot_curve \
/root/mmdetection/work_dirs/dcn_faster-rcnn_r101-dconv-c3-c5_fpn_1x_clp/20230815_081118/vis_data/20230815_081118.json \
/root/mmdetection/work_dirs/dcnv2_2044_200/20230817_013413/vis_data/scalars.json \
/root/mmdetection/work_dirs/swin_2044_200/20230819_040158/vis_data/20230819_040158.json \
/root/mmdetection/work_dirs/effb3_2044_200/20230818_072215/vis_data/20230818_072215.json \
--keys bbox_mAP \
--eval-interval 20 \
--legend dcn dcnv2 swin efficientNet --out results/result_mAP.png

# 테스트 셋  pkl 저장
python tools/test.py --out result/eff_2044_100.pkl configs/clp/effb3_fpn_8xb4-crop896-1x_clp.py work_dirs/effb3_2044_200/epoch_100.pth 
# dcn, dcnv2 마찬가지임 잘못학습되있음 

python tools/test.py --out result/dcn_2044_100.pkl configs/clp/dcn_faster-rcnn_r101-dconv-c3-c5_fpn_1x_clp.py work_dirs/dcn_faster-rcnn_r101-dconv-c3-c5_fpn_1x_clp/epoch_100.pth
python tools/test.py --out result/dcn2_2044_100.pkl configs/clp/dcnv2_clp.py work_dirs/dcnv2_2044_200/epoch_100.pth

python tools/analysis_tools/analyze_results.py \
       configs/clp/dcn_faster-rcnn_r101-dconv-c3-c5_fpn_1x_clp.py \
       result/eff_2044_100.pkl \
       results \
       --show