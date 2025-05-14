# 의생명정보학개론 I 프로젝트
팀원: 박정민, 이영주, 허지원

# 한국인의 식단 관리를 위한 챗봇 (A Dietary Management Chatbot for Koreans)

# Goal 
사용자로부터 음식 사진과 관련 텍스트를 입력으로 받고 이를 기반으로 영양정보/특징/장단점/질병과의 관련성을 제공하는 챗봇 서비스 개발

![image](https://github.com/user-attachments/assets/3fe64bd7-bfef-4234-8f84-58d6b1a2e17f)


# 한식 Data 전처리 (출처 : AIHUB)

	가) 데이터명

    		(1) 건강관리를 위한 음식 이미지
   
  	나) 데이터 통계
  
		    (1) 구조 : 이미지-JSON 쌍
		    
		    (2) 클래스 : 약 500개
		    
		    (3) 용량 : 약 900GB
		    
		    (4) 메타 정보 : 이미지명, 클래스명, BBox 좌표 등
![image](https://github.com/user-attachments/assets/6922152b-3911-4cb5-b829-585aaffb3bd3)

  다) 문제점
    (1) 용량 및 데이터 수가 너무 많아 학습 시간이 매우 오래걸림
    (2) 같은 클래스명 파일이 여러개 존재 (이미지 파일은 겹침 X)
    (3) 클래스명이 한글발음표기법 영어로 이루어져 직접 활용이 어려움
    (4) 클래스 간 데이터 수가 불균형함
    (5) 각 이미지의 해상도가 너무 크고, 불규칙적임
![image](https://github.com/user-attachments/assets/b5102f55-3537-4d38-85ca-471a25803919)

  라) 해결 방안
    (1) 직접 탐색해 얻은 데이터 통계를 엑셀 파일로 저장
    (2) 한글 클래스로 된 폴더명을 영어로 번역하여 저장
      (가) Googletrans 패키지 -> 번역 성능이 좋지 않음(직독직해, 유동적 X)
      (나) GPT-3.5-turbo -> 성능이 준수함(맥락 파악 O, 방대한 지식)
      (다) 데이터 통계를 정제(이상치, 결측치) 후 일관성을 위해 소문자 변환 적용
![image](https://github.com/user-attachments/assets/e03af950-fd08-4bfa-8a70-7e8137284dff)
    (3) 영어 클래스로 된 폴더 생성 및 대응되는 데이터를 이동 (JSON 파일 폐기)
    (4) Train set 기준 5000장 이상의 클래스 데이터 선별 
    (5) 선별된 255개의 클래스에 대하여 undersampling 기법 적용
    (6) 이미지 해상도를 224x224로 resize하여 저장 (Dataloader 단계 최적화)
![image](https://github.com/user-attachments/assets/ad97af32-8235-4fdb-87a4-431c81c7065f)

2) Glycemic Index(GI) Value 데이터 전처리 (출처 : 한국영양학회지)
  가) “한국인 상용 식품의 혈당지수 (Glycemic Index) 추정치를 활용한 한국 성인의         식사혈당지수 산출” 논문의 appendix 발췌
  나) 중요성이 떨어지는 데이터 제거
  다) “GI_glucose” 수치와 “GI_bread” 수치를 평균치로 통합 (모델의 혼돈 방지)
  라) “Food name” 항목 정제 (중복 클래스 통합 및 한식 데이터에 fitting)
![image](https://github.com/user-attachments/assets/9cbb1708-77c5-4c05-86eb-176301b1d6c0)

# CLIP-LoRA 전이학습 (Fine-tuning)
1) CLIP 
  가) 학습 데이터: 인터넷에서 4억 개 이미지와 해당 이미지에 대한 설명 Text를 pair
  나) 이미지와 텍스트를 인코더로 임베딩하여 같은 pair에 대해 거리를 가깝게 하고
            다른 pair에 대해 거리가 멀어지도록 학습

![image](https://github.com/user-attachments/assets/641dc6aa-d9f3-4537-bd82-00aa8748c849)

2) LoRA (Low-Rank Adaptation of Large Language Models)를 기존 CLIP 모델에 이식
  가) Pretrained weight를 freeze 하고, 어댑터 A와 B를 downstream task에 대해 학습

![image](https://github.com/user-attachments/assets/0fa61a21-c1b3-4dae-ba08-c8acf78e7483)

3) CLIP-LoRA 학습 흐름도
  가) 한식 classification에 대하여 fine-tuning 진행
  나) Text prompt: “A photo of a {class}” 
![image](https://github.com/user-attachments/assets/c397392d-75f2-46a4-8098-3cf78f90a294)

# CLIP-LoRA + LangChain + GPT 연결
1) CLIP-LoRA + FAISS 연결
![image](https://github.com/user-attachments/assets/2168da53-507f-40f3-952d-11e74c4de849)

2) GI 데이터(PDF) + Tool(ddg-search) + FAISS 연결
![image](https://github.com/user-attachments/assets/890047a8-973a-4f00-a802-153da20f39f5)

3) Agent + Memory + Conditioning Prompt
![image](https://github.com/user-attachments/assets/e12da896-e3ab-43db-9776-77429c179ffa)

# UI 제작 및 코드 통합
![image](https://github.com/user-attachments/assets/598dd0ed-3950-4300-9a63-cfa98009ab89)
![image](https://github.com/user-attachments/assets/4b2ba188-0d3b-4802-acc2-207799f169b4)

# 최종 모델 구조도
![image](https://github.com/user-attachments/assets/35f1ab98-c492-49f7-8373-2738f576f225)

# 결과
가. 실험 및 모델 평가
1) CLIP-LoRA 전이학습 모델 평가
![image](https://github.com/user-attachments/assets/e1f1c279-152b-43cb-beb8-866b2d279d09)

- 학습 전후 모델 성능 비교표
![image](https://github.com/user-attachments/assets/5cd93cba-9eeb-4061-bdac-235f9a6ceed2)



2) 최종 모델(챗봇) 평가
![image](https://github.com/user-attachments/assets/65eb5d61-7af2-4130-8386-aca7d3482e1e)
![image](https://github.com/user-attachments/assets/f99c86b1-30b6-4b75-bf4a-4f470eae3df8)
![image](https://github.com/user-attachments/assets/ae5330ed-7bec-4a73-b3cc-bfd9cb88c9c3)

# 추후 보완점
1) 기능적 한계
  가) 출력 답변 형식이 일관되지 않는 문제
  나) 원인 불명의 에러로 인한 실행 오류

2) 음식 클래스의 한계
  가) 학습 데이터 확장 및 재학습 필요
  나) 학습하지 않은 데이터 입력 시 대응하는 알고리즘 구현 필요

3) 사용자별 기간별 섭취 칼로리 계산 및 기억 기능 부재
  가) 음식의 양을 명확히 받는 기능 필요
  나) 축적 칼로리 계산 및 장기 기억 기능 필요


