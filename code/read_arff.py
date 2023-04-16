import arff
import traceback

try:
    with open('../data/WISDM_ar_v1.1_transformed.arff', 'r') as f:
        data = arff.load(f)
except Exception as e:
    print(traceback.format_exc())

# 길이 확인
print(len(data['data']))