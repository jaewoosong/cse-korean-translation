# 전진과 후진 (Forward and Backward)

전진과 후진은 신경망 계산의 필수 요소입니다.
(The forward and backward passes are the essential computations of a Net.)

<img src="fig/forward_backward.png" alt="Forward and Backward" width="480" />

단순한 로지스틱 회귀 분류기를 생각해 보겠습니다.
(Let’s consider a simple logistic regression classifier.)

전진 과정(forward pass)은 예측에 필요한 입력이 주어졌을 때에 출력을 계산합니다. 전진 과정에서 카페는 모델이 표현하는 "함수(function)"를 계산하기 위해 각 레이어의 계산을 합성합니다. 이 과정은 아래에서 위로 갑니다.
(The forward pass computes the output given the input for inference. In forward Caffe composes the computation of each layer to compute the "function" represented by the model. This pass goes from bottom to top.)

<img src="fig/forward.jpg" alt="Forward pass" width="320" />

데이터 x가 g(x)를 위해 벡터 내적 레이어를 통과하고 이어 h(g(x))를 위해 소프트맥스를 통과하며, f_W(x)를 위해 소프트맥스 손실을 통과합니다.
(The data x is passed through an inner product layer for g(x) then through a softmax for h(g(x)) and softmax loss to give f_W(x).)

후진 과정은 학습 손실이 주어졌을 때에 벡터 기울기를 계산합니다. 후진에서 카페는 자동 미분을 통해 각 레이어의 기울기를 역 합성하여 전체 모델의 기울기를 계산합니다. 이를 역전파(back-propagation)라고 합니다. 이 과정은 위에서 아래로 갑니다.
(The backward pass computes the gradient given the loss for learning. In backward Caffe reverse-composes the gradient of each layer to compute the gradient of the whole model by automatic differentiation. This is back-propagation. This pass goes from top to bottom.)

<img src="fig/backward.jpg" alt="Backward pass" width="320" />

후진 과정은 손실로 시작하며, 출력에 대한 기울기인 ∂f\_W/∂h가 계산됩니다. 모델의 나머지 부분에 대한 기울기는 레이어별로 미분의 연쇄 법칙을 통해 계산됩니다. 벡터 내적(`INNER_PRODUCT`) 레이어처럼 인자를 가진 레이어의 경우 그 인자에 대한 기울기인 ∂f\_W/∂W\_ip가 후진 단계에서 계산됩니다.
(The backward pass begins with the loss and computes the gradient with respect to the output ∂f\_W/∂h. The gradient with respect to the rest of the model is computed layer-by-layer through the chain rule. Layers with parameters, like the `INNER_PRODUCT` layer, compute the gradient with respect to their parameters ∂f\_W/∂W\_ip during the backward step.)

이 계산은 모델이 정의되자 마자 일어납니다. 카페는 전진과 후진 과정을 자동으로 계획하고 수행해 줍니다.
(These computations follow immediately from defining the model: Caffe plans and carries out the forward and backward passes for you.)

* `Net::Forward()`와 `Net::Backward()` 함수가 각각 전진과 후진 과정을 담당하며, `Layer::Forward()`와 `Layer::Backward()`가 각 단계에서 계산을 합니다.
(The `Net::Forward()` and `Net::Backward()` methods carry out the respective passes while `Layer::Forward()` and `Layer::Backward()` compute each step.)

* 각각의 레이어 종류는 `forward_{cpu,gpu}()`와 `backward_{cpu,gpu}()` 함수를 가지고 있어서 계산 설정에 따라 스스로의 진행 단계를 계산할 수 있습니다. 레이어는 제약 조건 혹은 편의를 위해 CPU나 GPU중 설정 중 하나만 구현할 수도 있습니다.
(Every layer type has `forward_{cpu,gpu}()` and `backward_{cpu,gpu}()` methods to compute its steps according to the mode of computation. A layer may only implement CPU or GPU mode due to constraints or convenience.)

연산기(the Solver)는 전진에서 출력과 손실을 얻어낸 뒤 후진에서 기울기를 계산하고, 그 후 손실을 줄이기 위한 가중치 갱신에 기울기를 사용하는 방식으로 모델을 최적화합니다. 연산기, 신경망, 레이어 간의 작업 분배는 카페를 모듈화 해 주며 카페를 사용한 개발을 쉽게 해 줍니다.
(The Solver optimizes a model by first calling forward to yield the output and loss, then calling backward to generate the gradient of the model, and then incorporating the gradient into a weight update that attempts to minimize the loss. Division of labor between the Solver, Net, and Layer keep Caffe modular and open to development.)

카페의 레이어 종류에 따른 전진과 후진에 대한 세부 설명은 레이어 부분에 대한 설명을 참조하세요.
(For the details of the forward and backward steps of Caffe’s layer types, refer to the layer catalogue.)
