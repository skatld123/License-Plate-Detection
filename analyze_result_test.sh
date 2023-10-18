python tools/test.py \
--out result/swin_2044_100.pkl \
configs/clp/swin_clp.py work_dirs/swin_2044_200/epoch_100.pth \
--show-dir /root/mmdetection/results/swin

python tools/test.py \
--out result/eff_2044_100.pkl \
configs/clp/effb3_fpn_8xb4-crop896-1x_clp.py work_dirs/effb3_2044_200/epoch_100.pth \
--show-dir /root/mmdetection/results/eff/ \

python tools/test.py \
--out result/dcnv2_2044_100.pkl \
configs/clp/dcnv2_clp.py \
work_dirs/dcnv2_2044_200/epoch_100.pth \
--show-dir /root/mmdetection/results/dcnv2/

python tools/test.py \
--out result/dcnv2_2044_100.pkl \
configs/clp/dcnv2_clp.py \
work_dirs/dcnv2_2044_200/epoch_100.pth \
--show-dir /root/mmdetection/results/dcnv2/