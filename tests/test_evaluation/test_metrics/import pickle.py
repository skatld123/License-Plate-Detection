import pickle

f = open('/root/mmdetection/work_dirs/result/dcnv2_detection/results_mega.pkl', 'rb')
info = pickle.load(f)
print(info)