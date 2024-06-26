# 이진 분류 - 2개의 클래스 중 하나를 고르기
# ? 도미 찾기

# 1. 가설 설정
# if fish_length > 30:
#     print("도미")

# 2. 데이터 준비
# 2-1. 도미 데이터 - 길이, 무게
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 2-2. 빙어 데이터 - 길이, 무게
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 3. 데이터 시각화
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

# 3-1. 도미 데이터 시각화
plt.scatter(bream_length, bream_weight, color='red', label='도미') # 빨간색으로 도미 데이터 표시

# 3-2. 빙어 데이터 시각화
plt.scatter(smelt_length, smelt_weight, color='blue', label='빙어') # 파란색으로 빙어 데이터 표시

# 그래프 설정
plt.title('이진 분류 - 도미일까, 빙어일까?') # 그래프 제목
plt.xlabel('생선의 길이') # x축: 길이
plt.ylabel('생선의 무게') # y축: 무게
plt.legend() # 범례 표시
# plt.show() # 그래프 출력

# 4. 최근접 이웃 알고리즘 도입
# 4-1. 도미와 빙어 데이터 병합
length = bream_length + smelt_length # 길이 데이터 병합
weight = bream_weight + smelt_weight # 무게 데이터 병합

# 4-2. 2차원 리스트로 변환
fish_data = [[l, w] for l, w in zip(length, weight)] # zip() 함수로 길이와 무게 데이터를 묶어 2차원 리스트로 변환

# 4-3. 타겟 데이터 생성 
# 곱셈 연산자(*)로 리스트를 반복하여 생성
fish_target = [1] * 35 + [0] * 14 # 도미 데이터: 35개, 빙어 데이터: 14개

# 4-4. 사이킷런 패키지 불러오기
from sklearn.neighbors import KNeighborsClassifier # from ~ import 구문을 사용하면 모듈 전체가 아닌, 일부만 불러올 수 있음

# 4-5. 객체 생성 및 훈련
kn = KNeighborsClassifier() # KNeighborsClassifier 객체 생성
kn.fit(fish_data, fish_target) # fit() 메서드로 훈련 데이터로 모델 훈련

# 4-6. 모델 평가
accuracy = kn.score(fish_data, fish_target)

# 4-7. 예측
predict_data = [20, 10]
predict = kn.predict([predict_data]) # predict() 메서드로 예측

# 그래프에 예측 데이터를 표시
plt.scatter(predict_data[0], predict_data[1], color='green', marker='^', label='예측') # 예측 데이터 표시
plt.legend() # 범례 표시

# 예측 결과 출력
# print(f"예측 결과: {'도미' if predict[0] == 1 else '빙어'}") # 삼항 연산자로 예측 결과 출력
# plt.show() # 그래프 출력

# 4-8. K 매개변수 확인
# get_params() 메서드로 객체의 매개변수 확인
params = kn.get_params()

# print(f"모델의 매개변수: {params}") # 모델의 매개변수: {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
# print(f"모델이 참조하는 K-이웃의 수: {params['n_neighbors']}") # n_neighbors 매개변수만 출력

# 4-9. K 매개변수 변경하고, K=49인 모델 생성
kn49 = KNeighborsClassifier(n_neighbors=49)

# K=49 모델 훈련
kn49.fit(fish_data, fish_target)

# K=49 모델 평가
accuracy49 = kn49.score(fish_data, fish_target)

# K=49 모델 예측
predict49 = kn49.predict([predict_data])

# 4-10. 예측 결과 출력 및 비교
# K=5 모델(기본값)
print(f"K=5 모델 예측 결과: {'도미' if predict[0] == 1 else '빙어'}") # K=5 모델 예측 결과: 빙어

# K=49 모델
print(f"K=49 모델 예측 결과: {'도미' if predict49[0] == 1 else '빙어'}") # K=49 모델 예측 결과: 도미
plt.show() # 그래프 출력