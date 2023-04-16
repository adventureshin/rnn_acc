import pandas as pd
import torch
import torch.nn as nn
from rnn_model import RNNModel
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from rnn_model import train, validate

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

# 훈련데이터와 테스스 데이터를 로드합니다.
train_data = pd.read_csv("../data/train_data.csv")
test_data = pd.read_csv("../data/test_data.csv")
# 칼럼 =  UNIQUE_ID,user,X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Z0,Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,XAVG,YAVG,ZAVG,XPEAK,YPEAK,ZPEAK,XABSOLDEV,YABSOLDEV,ZABSOLDEV,XSTANDDEV,YSTANDDEV,ZSTANDDEV,RESULTANT,class
# 입력 데이터 = X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Z0,Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9 를 (10,3) 형태로 변환합니다.
# 출력 데이터 = class 를 (1) 형태로 변환합니다.

selected_columns = [f'X{i}' for i in range(10)] + [f'Y{i}' for i in range(10)] + [f'Z{i}' for i in range(10)]
class_dict = {'Downstairs': 0, 'Jogging': 1, 'Sitting': 2, 'Standing': 3, 'Upstairs': 4, 'Walking': 5}

# class 를 숫자로 변환합니다.
train_data['class'] = train_data['class'].map(class_dict)
test_data['class'] = test_data['class'].map(class_dict)

# 상위 10개줄 출력
train_sequences = np.array([row[selected_columns].values.reshape(10, 3) for _, row in train_data.iterrows()])
test_sequences = np.array([row[selected_columns].values.reshape(10, 3) for _, row in test_data.iterrows()])
train_labels = np.array([row['class'] for _, row in train_data.iterrows()])
test_labels = np.array([row['class'] for _, row in test_data.iterrows()])

# 훈련 데이터와 테스트 데이터를 텐서로 변환합니다.
X_train = torch.tensor(train_sequences, dtype=torch.float32)
X_test = torch.tensor(test_sequences, dtype=torch.float32)

# 훈련 데이터와 테스트 데이터를 텐서로 변환합니다.
y_train = torch.tensor(train_labels, dtype=torch.long)
y_test = torch.tensor(test_labels, dtype=torch.long)

# 모델 초기화
input_size = 3
hidden_size_list = [64, 128, 256]
num_layers_list = [2, 3, 4, 5]
dropout_list = [0.2, 0.3, 0.5]
num_classes = 6
sequence_length = 10

# 훈련 및 검증 루프
num_epochs = 500
learning_rate_list = [0.01, 0.001, 0.0005]

batch_size_list = [32, 64, 128]
version = 1
# 성장 하지 않은 카운트
no_improve_count = 0

for i in range(30):
    with open("../train_history/rnn_acc_history.csv", "r") as f:
        # csv 행 개수 확인
        lines = f.readlines()
        if len(lines) == 1:
            n = 1
        else:
            n = int(lines[-1].split(",")[1]) + 1

    hidden_size = np.random.choice(hidden_size_list)
    num_layers = np.random.choice(num_layers_list)
    learning_rate = np.random.choice(learning_rate_list)
    batch_size = int(np.random.choice(batch_size_list))
    dropout = np.random.choice(dropout_list)
    model = RNNModel(input_size, hidden_size, num_layers, num_classes, sequence_length, device, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 버전 세팅
    version_n_time = f"{version}_{n}"
    print(
        f"version: {version_n_time}, hidden_size: {hidden_size}, num_layers: {num_layers}, learning_rate: {learning_rate}, batch_size: {batch_size}, dropout: {dropout}")
    # 데이터 로더 세팅
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_acc = 0
    loss_in_best_acc = 100
    before_train_acc = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # 훈련
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 검증
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # 가장 좋은 정확도 모델 저장
        if best_acc < test_acc:
            best_acc = test_acc
            loss_in_best_acc = test_loss
            torch.save(model.state_dict(), f"../rnn_model/{version_n_time}_rnn_model.pt")
        # 10번 연속으로 정확도가 향상되지 않으면 학습 종료, train_acc 가 1 이면 학습 종료
        if before_train_acc < train_acc:
            no_improve_count = 0
            before_train_acc = train_acc
        else:
            no_improve_count += 1

        # 10번 연속으로 정확도가 향상되지 않으면 학습 종료, train_acc 가 1 이면 학습 종료
        if no_improve_count == 10 or train_acc == 1:
            break

    # rnn_acc_history.csv 파일에 결과 추가
    with open(f"../train_history/rnn_acc_history.csv", "a") as f:
        f.write(
            f"\n{version}, {n}, {hidden_size}, {num_layers}, {learning_rate}, {batch_size}, {dropout},\
             {best_acc:.4f}, {loss_in_best_acc:.4f}")
        f.close()
