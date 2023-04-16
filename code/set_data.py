import pandas as pd
from io import StringIO

with open('../data/acceleration_data.txt', 'r') as file:
    lines = file.readlines()

# 데이터 정리 및 문제가 있는 줄 찾기
cleaned_lines = []
problematic_lines = []
for index, line in enumerate(lines):
    line_data = line.strip().split(',')

    # 7개의 값이 있는 경우 마지막 값을 제거
    if len(line_data) == 7:
        line_data.pop()

    if len(line_data) in [0, 1]:
        continue
    # 수정 후 데이터 개수가 여전히 6개가 아닌 경우 문제가 있는 줄로 간주
    elif len(line_data) != 6:
        # 두 개의 데이터가 합쳐진 경우 처리
        if len(line_data) == 11:
            # ';'로 구분된 데이터를 처리
            first_part = line_data[:5]
            second_part = line_data[5:]

            # 두 번째 데이터의 첫 번째 값에 있는 ';'를 기준으로 나눈 후 첫 번째 데이터의 끝에 값을 추가
            split_value = second_part[0].split(';')
            first_part.append(split_value[0])
            second_part[0] = split_value[1]

            cleaned_lines.append(','.join(first_part))
            cleaned_lines.append(','.join(second_part))
        else:
            print(len(line_data))
            problematic_lines.append((index, ','.join(line_data)))
    else:
        cleaned_lines.append(','.join(line_data))

# 문제가 있는 줄 출력
print("Problematic lines:")
for line_info in problematic_lines:
    print(f"Line {line_info[0] + 1}: {line_info[1]}")

# 정리된 데이터를 문자열로 저장
cleaned_data_str = '\n'.join(cleaned_lines)

# 문자열을 StringIO로 변환
cleaned_data_io = StringIO(cleaned_data_str)

# 정리된 데이터로 DataFrame 생성
ac_data = pd.read_csv(cleaned_data_io, header=None, sep=',')
ac_data.columns = ['age', 'action', 'time', 'x', 'y', 'z']
