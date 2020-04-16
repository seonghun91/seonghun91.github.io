
## CID: an efficient complexity-invariant distance for time series  

- 논문 링크 : https://www.researchgate.net/profile/Vinicius_Alves_de_Souza/publication/254861501_CID_An_efficient_complexity-invariant_distance_for_time_series/links/0deec51fe65bc20dfa000000/CID-An-efficient-complexity-invariant-distance-for-time-series.pdf

두 점 사이의 거리를 구하라고 하면 대부분 Euclidean Distance를 떠올릴것이다. 일부 시계열 데이터에서(특히 클러스터링의 경우) Euclidean Distance의 한계가 있는데 그점을 간단한 아이디어를 통해 해결하고자 한 논문이다. 구현 코드는 없으나 간단한 아이디어이므로 파이썬으로 직접 구현하였다.

#### 도입 배경
- 눈으로 보기에 유사해 보이는 두개의 object가 실제로는 더 거리차이가 날 수 있음  
- 특히, 복잡해 보이는 개체들은 단순한 개체에 더 assign 되는 경향이 있음.
    - 나의 비루한 영어 번역을 보완 설명하기 위해, 논문에서의 예시를 통해 설명하고자 한다.
    (사진1)
    - 위 그림(출처 : 본 논문 Fig.8 참조)을 설명하면, 도형의 중심부터 변까지의 거리를 각각 시계열 데이터로 풀어낸 그림이다. 변의 숫자가 늘어날수록, complexity도 늘어나는것을 알 수 있다.  
    (사진2)
    - 만약 여기서, 4번 도형과 가장 유사한 생김새의 도형은 무엇일까요? 라고 물으면 어떻게 대답할것인가? 아마도 5번이라고 대답할것이다. 마찬가지로, 32는? 이라는 질문에는 24로 대답을 할 것이고.
    - 그런데 각 도형의 시계열 데이터를 Euclidean Distance를 구한 위의 표를 보면(출처 : 본 논문 table 1 참조) 이상한 점들이 보인다, 32번 도형은 24보다도 4 도형과 더 가깝다는 결과가 나온다. 처음 도입 배경에서 복잡해 보이는 개체들이 단순한 개체에 더 assign되는 경향이 있다는 뜻이 바로 이말이다.  


#### (본 논문의) 접근 방안
- 이를 해결하기 위해 본 논문은 제목에서도 나와있는 CID(Complexity Invariant Distance)라는 개념을 제시하고 있다.
    (사진3)
    
- 위 식을 보면 시계열 Q, C에 대해 CID는 ED와 CF의 곱으로 나타낼 수 있다. 여기서 ED는 앞에서도 소개된 Euclidean Distance이고, 뒤의 CF가 바로 Complexity Correction Factor이다. 즉 기존 ED에 CF가 추가된 형태라고 볼 수 있다.
    (사진4)
- CF의 산출 방식은 다음과 같다. 각 시계열의 CE를 구해준 뒤, 위의 식에 대입하면 된다. 여기서 CE는 시계열의 complexity를 의미한다. 논문에서의 설명대로 쉽게 설명하면, 복잡한 시계열은 그렇지 않은 시계열에 비해 직선으로 늘렸을때 더 긴 길이를 가지게 될것이다.(위의 32번과 4번 도형을 직선으로 늘어뜨린다고 생각해보자.). 이를 계산하기 위해 diff 함수를 사용하여 계산할수 있으며, 해당 계산식을 파이썬으로 간단하게 구현하면 다음과 같다.


```python
import numpy as np
def complexityInvarianceDis(x, y) : # x, y는 시계열
    ce_x = np.sqrt(sum(np.diff(x) **2))
    ce_y = np.sqrt(sum(np.diff(y) **2))
    res = np.sqrt(sum((x - y)**2)) * max(ce_x, ce_y) / min(ce_x, ce_y)
    
    return res
```

- 아무래도 코드를 보면 좀 더 직관적으로 이해할 수 있다. 시계열 x의 복잡도와 y의 복잡도가 거의 같다면, max(ce_x, ce_y) / min(ce_x, ce_y) 값은 1에 가까울 것이고, 이때는 ED와 비슷함을 알 수 있다.
- 반대로 둘의 차이가 크다면, max(ce_x, ce_y) / min(ce_x, ce_y)의 크기는 커질것이고, ED에 앞의 값이 곱해져서 값이 더욱 커짐을 알 수있다. 즉, 복잡성의 차이가 커질수록 둘 사이의 거리가 더 벌어지는 구조이다.

#### 후기
- 두 점 사이의 거리를 재는데는 무의식적으로 Euclidean Distance를 써왔는데, 문제 제기와 이를 아주 간단한 아이디어로 풀어냈다는 점이 매우 신기했으며, 변수 클러스터링 할일이 있을 때 지금 이 접근방식을 꼭 실험해보도록 하자.  
- 클러스터링 외에 이 개념을 더 활용할 수 있는 분야는 없을까? 두개의 시계열을 비교하는 경우 외에도, 한 시계열 내에서도 복잡성을 측정하고, 이를 활용해볼만한 분야를 찾아보자


```python

```
