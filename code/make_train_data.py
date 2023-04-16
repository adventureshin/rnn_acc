import pandas as pd
import arff
import traceback

# 데이터 로드
try:
    with open('../data/WISDM_ar_v1.1_transformed.arff', 'r') as f:
        data = arff.load(f)
except Exception as e:
    print(traceback.format_exc())

# 데이터프레임으로 변환
df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

# 데이터 섞기
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# train_data와 test_data로 나누기 (80%: train, 20%: test)
split_index = int(0.8 * len(df))
train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

# train_data와 test_data를 각각 CSV 파일로 저장
train_data.to_csv('../data/train_data.csv', index=False)
test_data.to_csv('../data/test_data.csv', index=False)
