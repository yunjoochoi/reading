1. adopt the ODE-to-SDE strategy to overcome the deterministic nature of the original flow model

2. 이미지 생성 시 사용되는 스텝을 줄여 샘플링 속도를 빠르게 함 (트레이닝 타임스텝을 10으로 두고도 , 디노이징 스텝임, 효과적. 인퍼런스는 40스텝(오리지널 sd3 설정))

- ls that this approach enables fast training without sacrificing image quality at test time

3. Kullback-Leibler (KL) constrain 사용하여 리워드 해킹 일어나지 않게 함 (원래 grpo에 kl 텀 있긴 함)


grpo는 크리틱 없애고 그룹 내에서의 상대평가 도입함: Advantage = 그룹 내에서 평균보다 높은지 낮은지 정규화함.

그룹 사이즈: 	
When the group size was reduced to G = 12 and G = 6, training became unstable and eventually collapsed, whereas G = 24 remained stable throughout the process
