# 손실 (Loss)

다른 대부분의 기계 학습에서와 같이 카페에서는 학습이 손실 함수(에러, 비용, 대상 함수 등으로도 표현됨)에 의해 주도됩니다. 손실 함수는 인자 설정값(현재 신경망 가중치)을 그 인자 설정값이 얼마나 "나쁜지"를 나타내는 스칼라 값에 연결함으로써 학습의 목표를 나타냅니다. 따라서 학습의 목표는 손실 함수를 최소화하는 가중치를 찾는 일이 됩니다.
(In Caffe, as in most of machine learning, learning is driven by a loss function (also known as an error, cost, or objective function). A loss function specifies the goal of learning by mapping parameter settings (i.e., the current network weights) to a scalar value specifying the "badness" of these parameter settings. Hence, the goal of learning is to find a setting of the weights that _minimizes_ the loss function.)

카페의 손실은 신경망의 전진에 의해 계산됩니다. 각각의 레이어는 (`아래의`) 블롭에서 입력을 받아서 (`위의`) 블롭으로 출력을 내보냅니다. 몇몇 레이어의 출력이 손실 함수에 사용될 수 있습니다. '하나 대 전부'의 분류 작업을 위한 일반적인 손실 함수 선택은 `SoftmaxWithLoss` (손실 지원 소프트맥스) 함수이며, 신경망 정의에서 다음과 같이 사용됩니다.
(The loss in Caffe is computed by the Forward pass of the network. Each layer takes a set of input (`bottom`) blobs and produces a set of output (`top`) blobs. Some of these layers’ outputs may be used in the loss function. A typical choice of loss function for one-versus-all classification tasks is the `SoftmaxWithLoss` function, used in a network definition as follows, for example:)

    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "pred"
      bottom: "label"
      top: "loss"
    }

`SoftmaxWithLoss` 함수에서 `위` 블롭은 전체 미니배치에 대한 손실(예측 결과인 `pred`와 실제 값인 `label`로 계산되는)의 평균값이며 스칼라(모양이 없는)입니다.
(In a `SoftmaxWithLoss` function, the `top` blob is a scalar (empty shape) which averages the loss (computed from predicted labels `pred` and actuals labels `label`) over the entire mini-batch.)

## 손실 가중치 (Loss weights)

다중 레이어가 손실을 계산하는 신경망의 경우 (예: 입력값을 분류하기 위해 `SoftmaxWithLoss` 레이어를 사용하고 그것을 다시 `EuclideanLoss` (유클리드 손실) 레이어로 재구성하는 경우) 손실간의 상대적 중요도를 나타내기 위해 가중치를 사용할 수 있습니다.
(For nets with multiple layers producing a loss (e.g., a network that both classifies the input using a `SoftmaxWithLoss` layer and reconstructs it using a `EuclideanLoss` layer), _loss weights_ can be used to specify their relative importance.)

일반적으로 카페의 레이어 중 이름이 `Loss`로 끝나는 것은 손실 함수에 사용될 수 있지만 다른 레이어들은 순수하게 중간 연산에만 사용된다고 가정됩니다. 하지만 `loss_weight: <float>`이라는 항목이 레이어가 생성하는 `위` 블롭에 대한 정의에 추가되면 어떤 레이어라도 손실 계산에 사용될 수 있습니다. 이름이 `Loss`로 끝나는 레이어는 내부적으로 `loss_weight: 1` 항목을 첫 `위` 블롭에 가지고 있습니다(그리고 다른 추가 `top` 블롭에는 `loss_weight: 0`이라고 적혀 있습니다.). 다른 레이어들은 내부적으로 `loss_weight: 0`을 모든 `위` 블롭에 대해 가지고 있습니다. 따라서 위의 `SoftmaxWithLoss` 레이어를 다음과 같이 작성해도 똑같습니다.
(By convention, Caffe layer types with the suffix `Loss` contribute to the loss function, but other layers are assumed to be purely used for intermediate computations. However, any layer can be used as a loss by adding a field `loss_weight: <float>` to a layer definition for each `top` blob produced by the layer. Layers with the suffix `Loss` have an implicit `loss_weight: 1` for the first `top` blob (and `loss_weight: 0` for any additional `top`s); other layers have an implicit `loss_weight: 0` for all `top`s. So, the above `SoftmaxWithLoss` layer could be equivalently written as:)

    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "pred"
      bottom: "label"
      top: "loss"
      loss_weight: 1
    }

하지만 역전파가 가능한 모든 레이어에 0이 아닌 `loss_weight`가 주어져야만, 예를 들자면 필요한 경우 신경망의 중간 레이어들이 만들어낸 활성(activation)을 정형화(regularize)하는 것이 가능해집니다. 손실이 0이 아닌 비단독(non-singleton) 출력의 경우 손실은 단순히 블롭의 모든 항목을 더하는 것으로 계산됩니다.
(However, any layer able to backpropagate may be given a non-zero `loss_weight`, allowing one to, for example, regularize the activations produced by some intermediate layer(s) of the network if desired. For non-singleton outputs with an associated non-zero loss, the loss is computed simply by summing over all entries of the blob.)

카페에서의 최종 손실은 결국 다음의 대략 코드(pseudo-code)와 같이 신경망 전체에 걸친 손실 가중치를 합치는 연산입니다.
(The final loss in Caffe, then, is computed by summing the total weighted loss over the network, as in the following pseudo-code:))

    loss := 0
    for layer in layers:
      for top, loss_weight in layer.tops, layer.loss_weights:
        loss += loss_weight * sum(top)
