---
layout: post
title:  "[논문] Distributed and parallel time series feature extraction for industrial big data applications"
subtitle:   ""
categories: TimeSeries
#tags: 시계열
comments: true

---


- 논문 링크 : https://arxiv.org/pdf/1610.07717v3.pdf
- 코드 : https://github.com/blue-yonder/tsfresh

 시계열 데이터의 feature 추출에 대한 논문이다. 이 논문을 첫 번째로 선택한 이유는, 아무래도 paperswithcodes의 time series 도메인 중 인기가 높았으며, 코드 또한 잘 구현되어 있기 때문이다. 해당 논문을 간략하게 요약하면 다음과 같다.

 #### 도입 배경
 - 머신 러닝 문제에서 feature selection은 늘 중요하게 다뤄지고 있는 문제이며, 특히 시계열 데이터셋인 경우 더욱 어려움  
 - Boruta, LDA, DTW 등등의 방식을 통해 Feature들을 다루고 있음

 #### (본 논문의) 접근 방안
 - 1) 전체 변수들을 대상으로 파생 feature들을 생성(min, max, std 등)
 - 2) 생성된 전체 변수들을 한개씩 Target과 Hypothsis test 진행하여 변수별 p-value 생성
 - 3) feature significance test를 통해 최종 변수 선별  
     - FDR(False Discovery Rate)를 잡기 위한 과정
     - Benjamini-Yekutieli procedure를 통해 기준선 이하의 p-value를 가진 변수들만 선택
     
 #### 실험 결과  
 - boruta, LDA, DTW 등과 비교했을때 준수한 accuracy와 std값을 보이며 수행시간도 적었음 


 #### 구현 코드 분석
 - 논문을 tsfresh라는 파이썬 패키지로 구현하고 있는데, 아래 코드를 통해서 해당 접근 방식을 파악해보았다.


```python
import warnings
warnings.filterwarnings('ignore')
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()
#timeseries.head()
```

<img width="288" alt="스크린샷 2020-04-10 오후 7 48 04" src="https://user-images.githubusercontent.com/28383546/78986582-89fa8a00-7b66-11ea-96b3-4b903f8fc745.png">


친절하게도 패키지 내에 사용할만한 예제를 첨부해 주었다. 예제가 되는 timeseries 데이터는 id(88개) 별로 time이 0~15를 가지는 데이터이며,
그 결과값은 y에 binary로 저장되어 있다.

기계 오작동을 판별하는 데이터라고 예를 들면,  y가 고장 여부(0 또는 1)이라고 하면 id는 부품, F_x부터의 변수들은 부품의 각 센서 값이라고 이해하면 좋을듯하다.


```python
from tsfresh import extract_features
extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
#extracted_features.head()
```

    Feature Extraction: 100%|██████████| 20/20 [00:07<00:00,  2.90it/s]

<img width="695" alt="스크린샷 2020-04-10 오후 7 48 33" src="https://user-images.githubusercontent.com/28383546/78986628-a3033b00-7b66-11ea-9d63-e219a2889ee7.png">


앞서 논문 중 접근방안의 '1)'단계인 파생변수들을 생성하는 단계이다.  
위의 표와 같이 6개의 변수들에 대해 무려 4536개의 feature들을 생성했다.  

파생변수가 어떻게 생성되었는지를 확인하기 위해 extract_features 함수 내부를 보면,


```python
df_melt, column_id, column_kind, column_value = \
        dataframe_functions._normalize_input_to_internal_representation(
            timeseries_container=timeseries_container,
            column_id=column_id, column_kind=column_kind,
            column_sort=column_sort,
            column_value=column_value)
    # Use the standard setting if the user did not supply ones himself.
    if default_fc_parameters is None and kind_to_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()
    elif default_fc_parameters is None and kind_to_fc_parameters is not None:
        default_fc_parameters = {}

    # If requested, do profiling (advanced feature)
    if profile:
        profiler = profiling.start_profiling()

    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter("ignore")
        else:
            warnings.simplefilter("default")

        result = _do_extraction(df=df_melt,
                                column_id=column_id, column_value=column_value,
                                column_kind=column_kind,
                                n_jobs=n_jobs, chunk_size=chunksize,
                                disable_progressbar=disable_progressbar,
                                show_warnings=show_warnings,
                                default_fc_parameters=default_fc_parameters,
                                kind_to_fc_parameters=kind_to_fc_parameters,
                                distributor=distributor)
```

함수의 핵심 부분만 발췌한 결과값이다. 먼저 데이터를 melt하여 저장하고,
ComprehensiveFCParameters 파라미터를 통해 어떤 파생변수들을(min, max, lag emd) 만들것인지 정해준 뒤,
마지막으로 do_extraction을 통해 피쳐 작업을 한다.
참고로 do_extraction이 실행되면 병렬 작업이 수행되며 실제로는 do_extraction_on_chunk 함수를 통해 연산이 이뤄진다.


```python
name_to_param.update({
            "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
            "c3": [{"lag": lag} for lag in range(1, 4)],
            "cid_ce": [{"normalize": True}, {"normalize": False}],
            "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
            "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
            "quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
            "autocorrelation": [{"lag": lag} for lag in range(10)],
            "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
        })
```

ComprehensiveFCParameters의 함수 내부를 일부분만 발췌한 결과이다. 
correaltion, lag, min, max 등 다양한 feature들을 가지고 있으며, 해당 값들이 feature_calculator 함수를 통해서 계산이 된다.


```python
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, y)
#features_filtered.head()
```

<img width="772" alt="스크린샷 2020-04-10 오후 7 49 06" src="https://user-images.githubusercontent.com/28383546/78986645-b0202a00-7b66-11ea-9053-471acfef1729.png">

앞서 변수들을 생성했다면, 변수 선별 작업들을 진행한다.  
select_features 함수를 통해 최종적으로 627개의 변수들이 최종 선별 되었다.  
마찬가지로 select_features 함수도 내부적으로 확인을 해보면,


```python
relevance_table = calculate_relevance_table(
        X, y, ml_task=ml_task, n_jobs=n_jobs, show_warnings=show_warnings, chunksize=chunksize,
        test_for_binary_target_real_feature=test_for_binary_target_real_feature,
        fdr_level=fdr_level, hypotheses_independent=hypotheses_independent,
    )
```

calculate_relevance_table 함수가 최종적으로 돌아가며, 이 함수에서는
변수-Target이 범주형/수치형에 따른 4가지 조합에 대한 hypothesis test를 수행하고,  
최종적으로 변수를 선별하는 작업을 진행한다.  

#### 개인적인 후기   
모델링을 많이 해본 사람들은 공감하겠지만, feature engineering 과정은 매우 중요하나    
그 과정은 매우 귀찮고 힘든?과정이다... 특히 해당 도메인을 모르는 경우 더욱...그런 의미에서 이 논문이 신선했던 부분은 다음과 같다.  
- 1) 다양한 시계열 feature들을 손쉽게 생성해 줄 수 있어, 초기 baseline을 잡거나 퍼포먼스 향상에 큰 도움이 될 수 있는 점. 데이터가 주어졌을때, 빠르게 baseline 모델을 잡아 어떤 변수들이 유효했는지를 역으로 탐구하고, 그 결과 데이터의 내부를 확인하는데도 쓸 수 있을듯?
- 2) 단순히 Target-변수간의 p-value가 유의(0.05, 0.01)한 값만을 선별한 것이 아닌 FDR을 잡을 수 있도록 설계하여 선별된 변수들이 실제적으로 유의한 값들이 선별될 수 있게한 점. 개인적으로는 hypothesis test 한 코드들만 드러내서 초기 EDA에 사용해도 매우 유용할것 같다는 생각도 든다..

하지만 아쉬운 점도 있었는데(내가 뭐라고...) 아쉬운 점들을 적어보면 다음과 같다.
- 1) 특정 시계열 데이터에서만 적용 가능함. 위의 예에서도 보겠지만 id 별로 동일한 시간의 나열된 데이터일때 계산이 가능. 즉 고객별로 초기 유입 시점이 다르거나, 고객별로 target값이 변하는 데이터의 경우 적용하기가 힘들다.
- 2) 변수 선별단계에서 p-value가 낮으며 상관관계가 높은 변수들이 많다면 선별되는 변수들의 질이 떨어질 수 있는점?
    - 이 점은 논문에서도 변수 선별 전, 또는 후에 pca를 권장하긴 했다. pca를 그렇게 좋아하진 않기에.. 해당 부분이 우려된다면 초기 변수를 어느정도 다뤄주는 작업이 필요할 것 같다. 

#### 더 공부할 논문 / 할일
- LDA 논문(B. D. Fulcher, N. S. Jones, Highly Comparative Feature-Based Time-Series Classification, Knowledge and Data Engineering, IEEE Transactions on 26 (12) (2014) 3026–3037.)
- Boruta 논문 (M. B. Kursa, W. R. Rudnicki, The all relevant feature selection using random forest, arXiv preprint arXiv:1106.5112 .)
- DTW 논문 ( S. Van Der Walt, S. C. Colbert, G. Varoquaux, The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering 13 (2) (2011) 22–30.)
- 패키지를 현재 참가중인 kaggle 대회에 적용해보기
- 이 패키지에서 사용된 time series feature들을 하나씩 뜯어보고 향후 활용할 수 있도록 체화하기
- 깔끔하게 만든 패키지를 뜯어보니 나도 나만의 패키지를 만들고 싶....~~지만 매우 고되고 힘든 작업이겠지~~
