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

(For nets with multiple layers producing a loss (e.g., a network that both classifies the input using a SoftmaxWithLoss layer and reconstructs it using a EuclideanLoss layer), loss weights can be used to specify their relative importance.)

(By convention, Caffe layer types with the suffix Loss contribute to the loss function, but other layers are assumed to be purely used for intermediate computations. However, any layer can be used as a loss by adding a field loss_weight: <float> to a layer definition for each top blob produced by the layer. Layers with the suffix Loss have an implicit loss_weight: 1 for the first top blob (and loss_weight: 0 for any additional tops); other layers have an implicit loss_weight: 0 for all tops. So, the above SoftmaxWithLoss layer could be equivalently written as:)

    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "pred"
      bottom: "label"
      top: "loss"
      loss_weight: 1
    }

(However, any layer able to backpropagate may be given a non-zero loss_weight, allowing one to, for example, regularize the activations produced by some intermediate layer(s) of the network if desired. For non-singleton outputs with an associated non-zero loss, the loss is computed simply by summing over all entries of the blob.)

(The final loss in Caffe, then, is computed by summing the total weighted loss over the network, as in the following pseudo-code:)

    loss := 0
    for layer in layers:
      for top, loss_weight in layer.tops, layer.loss_weights:
        loss += loss_weight * sum(top)
