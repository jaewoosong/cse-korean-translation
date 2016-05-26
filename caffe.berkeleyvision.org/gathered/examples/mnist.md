# 카페로 LeNet을 MNIST에 대해 학습시키기 (Training LeNet on MNIST with Caffe)

카페가 성공적으로 컴파일 되었다고 가정하겠습니다. 그렇지 않다면 [설치 페이지]()를 참조해 주세요. 이 강좌에서는 카페가 `CAFFE_ROOT`에 있다고 가정합니다.
(We will assume that you have Caffe successfully compiled. If not, please refer to the [Installation page](). In this tutorial, we will assume that your Caffe installation is located at `CAFFE_ROOT`.)

## 데이터셋 준비 (Prepare Datasets)

먼저 MNIST 웹사이트에서 데이터를 받고 형식을 변환해야 합니다. 간단히 다음의 명령어를 입력하세요.
(You will first need to download and convert the data format from the MNIST website. To do this, simply run the following commands:)

    cd $CAFFE_ROOT
    ./data/mnist/get_mnist.sh
    ./examples/mnist/create_mnist.sh

만약에 `wget`이나 `gunzip`이 설치되지 않았다고 불평하는 메시지가 나오면 그 프로그램들을 설치해 주세요. 위의 스크립트를 실행하면 `mnist_train_lmdb`와 `mnist_test_lmdb` 두 데이터셋이 생겨야 합니다.
(If it complains that `wget` or `gunzip` are not installed, you need to install them respectively. After running the script there should be two datasets, `mnist_train_lmdb`, and `mnist_test_lmdb`.)

## LeNet: MNIST 분류 모델 (LeNet: the MNIST Classification Model)

학습 프로그램을 실행하기 전에 설명을 먼저 하도록 하겠습니다. 숫자 분류 작업에 뛰어나다고 알려진 LeNet 신경망을 사용할 것입니다. 뉴런의 활성화를 위해 시그모이드 활성 대신 수정된 선형 단위(ReLU) 활성을 사용했다는 점이 원본 LeNet의 구현과 살짝 다른 점입니다.
(Before we actually run the training program, let’s explain what will happen. We will use the LeNet network, which is known to work well on digit classification tasks. We will use a slightly different version from the original LeNet implementation, replacing the sigmoid activations with Rectified Linear Unit (ReLU) activations for the neurons.)

LeNet의 설계는 이미지넷 등에 적용되는 더 큰 신경망에서도 아직도 사용되는 합성곱 신경망(CNN)의 핵심을 담고 있습니다. 크게 보자면 LeNet은 합성곱 레이어와 그에 뒤따라 오는 통합(pooling) 레이어, 또다시 합성곱 레이어와 통합 레이어, 그 다음에 평범한 다중 레이어 퍼셉트론과 비슷하게 상대방의 모든 뉴런과 서로 연결된 두 레이어(fully connected layers)로 이루어져 있습니다. 레이어들은 `$CAFFE_ROOT/examples/mnist/lenet_train_test.prototxt`에 정의되어 있습니다.
(The design of LeNet contains the essence of CNNs that are still used in larger models such as the ones in ImageNet. In general, it consists of a convolutional layer followed by a pooling layer, another convolution layer followed by a pooling layer, and then two fully connected layers similar to the conventional multilayer perceptrons. We have defined the layers in `$CAFFE_ROOT/examples/mnist/lenet_train_test.prototxt`.)

## MNIST 신경망 정의하기 (Define the MNIST Network)

이 단원은 MNIST 손으로 쓴 숫자 분류를 위한 LeNet 모델을 정의한 `lenet_train_test.prototxt`에 있는 모델 정의를 설명합니다. 여러분이 구글 프로토버프에 익숙하다고 가정하며, `$CAFFE_ROOT/src/caffe/proto/caffe.proto`에 적혀있는 카페에서 사용하는 프로토버프에 대한 정의를 읽어 보았다고 가정합니다.
(This section explains the `lenet_train_test.prototxt` model definition that specifies the LeNet model for MNIST handwritten digit classification. We assume that you are familiar with Google Protobuf, and assume that you have read the protobuf definitions used by Caffe, which can be found at `$CAFFE_ROOT/src/caffe/proto/caffe.proto`.)

자세히 말하자면, 우리는 `caffe::NetParameter` (혹은 파이썬에서라면 `caffe.proto.caffe_pb2.NetParameter`) 프로토버프를 작성할 것입니다. 먼저 신경망에 이름을 붙이는 데에서부터 시작합니다.
(Specifically, we will write a `caffe::NetParameter` (or in python, `caffe.proto.caffe_pb2.NetParameter`) protobuf. We will start by giving the network a name:)

    name: "LeNet"
    
### 데이터 레이어 작성 (Writing the Data Layer)

지금 우리는 이 예제의 앞 부분에서 우리가 만든 lmdb에서 MNIST 데이터를 불러 올 것입니다. 이 작업은 데이터 레이어에 정의되어 있습니다.
(Currently, we will read the MNIST data from the lmdb we created earlier in the demo. This is defined by a data layer:)

    layer {
      name: "mnist"
      type: "Data"
      transform_param {
        scale: 0.00390625
      }
      data_param {
        source: "mnist_train_lmdb"
        backend: LMDB
        batch_size: 64
      }
      top: "data"
      top: "label"
    }

자세히 보자면, 이 레이어는 이름이 `mnist`, 종류가 `data`이고 주어진 lmdb 파일로부터 데이터를 읽는 작업을 합니다. 일괄 처리 크기는 64이고, 입력되는 픽셀을 조정해서 픽셀 값이 [0,1) (0 이상 1 미만)이 되게 합니다. 0.00390625은 무엇일까요? 1 나누기 256입니다. 마지막으로 이 제이어는 두 개의 블롭, `data` 블롭과 `label` 블롭을 생성합니다.
(Specifically, this layer has name `mnist`, type `data`, and it reads the data from the given lmdb source. We will use a batch size of 64, and scale the incoming pixels so that they are in the range [0,1). Why 0.00390625? It is 1 divided by 256. And finally, this layer produces two blobs, one is the `data` blob, and one is the `label` blob.)

### 합성곱 신경망 작성하기 (Writing the Convolution Layer)

첫 번째 합성곱 레이어를 정의합시다.
(Let’s define the first convolution layer:)

    layer {
      name: "conv1"
      type: "Convolution"
      param { lr_mult: 1 }
      param { lr_mult: 2 }
      convolution_param {
        num_output: 20
        kernel_size: 5
        stride: 1
        weight_filler {
      type: "xavier"
        }
        bias_filler {
      type: "constant"
        }
      }
      bottom: "data"
      top: "conv1"
    }

이 레이어는 `data` 블롭(데이터 레이어로부터 제공된)을 받아서 `conv1` 레이어를 만듭니다. 이 레이어는 출력 채널이 20개, 출력 합성곱 커널(알맹이라는 뜻) 크기가 5이고 실행될 때의 뜀뛰기(stride) 값은 1입니다.
(This layer takes the `data` blob (it is provided by the data layer), and produces the `conv1` layer. It produces outputs of 20 channels, with the convolutional kernel size 5 and carried out with stride 1.)

충전재(filler)를 통해 무작위로 가중치와 편향치(bias)를 초기화할 수 있습니다. 가중치 충전재(weight filler)의 경우 입력과 출력 뉴런의 개수에 근거하여 자동으로 초기화 시 크기 조절량을 결정해주는 `xavier` 알고리즘을 사용합니다. 편향치 충전재(bias filler)의 경우 간단하게 상수로 초기화하며 그 경우 기본값은 0입니다.
(The fillers allow us to randomly initialize the value of the weights and bias. For the weight filler, we will use the `xavier` algorithm that automatically determines the scale of initialization based on the number of input and output neurons. For the bias filler, we will simply initialize it as constant, with the default filling value 0.)

`lr_mults`는 레이어의 학습 가능한 인자들에 대한 학습률 조정치입니다. 이번 경우 가중치 학습률을 연산기가 실행 시 만들어내는 학습률과 같게 되도록 설정하였고, 편향치 학습률은 그것의 두 배가 되도록 설정하였습니다 - 이 설정이 종종 좋은 수렴률을 보여줍니다.
(`lr_mult`s are the learning rate adjustments for the layer’s learnable parameters. In this case, we will set the weight learning rate to be the same as the learning rate given by the solver during runtime, and the bias learning rate to be twice as large as that - this usually leads to better convergence rates.)

### 통합 레이어 작성하기 (Writing the Pooling Layer)

휴우~. 사실 통합 레이어는 만들기가 훨씬 쉽습니다.
(Phew. Pooling layers are actually much easier to define:)

    layer {
      name: "pool1"
      type: "Pooling"
      pooling_param {
        kernel_size: 2
        stride: 2
        pool: MAX
      }
      bottom: "conv1"
      top: "pool1"
    }

크기 2의 통합 커널로 2씩 뜀뛰기(stride)를 해서 (근접한 통합 영역끼리 겹치는 부분이 없도록) 최대치로 통합(max pooling)을 한다는 뜻입니다.
(This says we will perform max pooling with a pool kernel size 2 and a stride of 2 (so no overlapping between neighboring pooling regions).)

비슷하게 두 번째 합성곱과 통합 레이어를 만들 수 있습니다. 자세한 내용은 `$CAFFE_ROOT/examples/mnist/lenet_train_test.prototxt`를 참조하세요.
(Similarly, you can write up the second convolution and pooling layers. Check `$CAFFE_ROOT/examples/mnist/lenet_train_test.prototxt` for details.)

### 모두 연결된 레이어 만들기 (Writing the Fully Connected Layer)

모두 연결된 레이어를 만드는 것 또한 매우 간단합니다.
(Writing a fully connected layer is also simple:)

    layer {
      name: "ip1"
      type: "InnerProduct"
      param { lr_mult: 1 }
      param { lr_mult: 2 }
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "pool2"
      top: "ip1"
    }

500개의 출력을 만드는 모두 연결된 레이어(카페에서는 `내적`(`InnerProduct`) 레이어라고 부릅니다)입니다. 다른 내용은 다 눈에 익지요?
(This defines a fully connected layer (known in Caffe as an `InnerProduct` layer) with 500 outputs. All other lines look familiar, right?)

### 정류된 선형 단위(ReLU) 레이어 만들기 (Writing the ReLU Layer)

정류된 선형 단위(정선단, ReLU) 레이어를 만드는 것도 쉽습니다.
(A ReLU Layer is also simple:)

    layer {
      name: "relu1"
      type: "ReLU"
      bottom: "ip1"
      top: "ip1"
    }

정선단은 원소에 대한 연산이기 때문에 추가 변수를 사용하지 않음으로써 메모리를 절약할 수 있습니다. 단순히 아래와 위 블롭 이름을 똑같이 해 주는 것만으로 가능합니다. 물론 다른 레이어 종류에서는 블롭 이름을 중복해서 쓰면 안 됩니다!
(Since ReLU is an element-wise operation, we can do _in-place_ operations to save some memory. This is achieved by simply giving the same name to the bottom and top blobs. Of course, do NOT use duplicated blob names for other layer types!)

정선단 레이어 다음에는 또다른 모두 연결된 레이어를 만듭니다 ^^
(After the ReLU layer, we will write another innerproduct layer:))

    layer {
      name: "ip2"
      type: "InnerProduct"
      param { lr_mult: 1 }
      param { lr_mult: 2 }
      inner_product_param {
        num_output: 10
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      bottom: "ip1"
      top: "ip2"
    }

### 손실 레이어 만들기 (Writing the Loss Layer)

드디어 손실을 계산합니다!
(Finally, we will write the loss!)

    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "ip2"
      bottom: "label"
    }

소프트맥스 손실(`softmax_loss`) 레이어는 (시간을 절약하고 계산의 안정성을 향상시키도록) 소프트맥스와 다변수 로지스틱 손실 모두를 구현하고 있습니다. 두 개의 블롭, 예측값과 데이터 레이어(기억나죠?)로부터 나온 라벨(`label`)이 사용됩니다. 출력을 따로 만들지는 않습니다. 이 레이어가 하는 일은 `ip2`에 대해서 손실 함수를 계산해서 역전파가 시작될 때에 그 값을 보고하고 경사 하강법을 개시하는 것입니다. 여기가 모든 마법이 시작되는 지점입니다.
(The `softmax_loss` layer implements both the softmax and the multinomial logistic loss (that saves time and improves numerical stability). It takes two blobs, the first one being the prediction and the second one being the `label` provided by the data layer (remember it?). It does not produce any outputs - all it does is to compute the loss function value, report it when backpropagation starts, and initiates the gradient with respect to `ip2`. This is where all magic starts.)

### 추가: 레이어 작성 법칙 (Additional Notes: Writing Layer Rules)
아래와 같이, 레이어 정의는 레이어가 신경망 정의에 포함될 것인지, 포함된다면 언제 포함될 것인지에 대한 규칙을 포함할 수 있습니다.
(Layer definitions can include rules for whether and when they are included in the network definition, like the one below:)

    layer {
      // ...layer definition...
      include: { phase: TRAIN }
    }

레이어가 신경망에 포함될 것인지에 대한 이 규칙은 신경망의 현재 상태에 기반합니다. `$CAFFE_ROOT/src/caffe/proto/caffe.proto`에서 레이어 규칙과 모델 스키마에 대한 더 많은 정보를 찾을 수 있습니다.
(This is a rule, which controls layer inclusion in the network, based on current network’s state. You can refer to `$CAFFE_ROOT/src/caffe/proto/caffe.proto` for more information about layer rules and model schema.)

위의 예제 레이어는 학습(`TRAIN`) 단계에서만 사용됩니다. 만약 `TRAIN`을 `TEST`로 바꾸면 이 레이어는 평가 단계에서만 사용되게 됩니다. 기본적으로 레이어 규칙이 없을 경우 레이어는 모든 신경망에 다 사용되게 됩니다. 따라서 `lenet_train_test.prototxt`는 두 개의 (`batch_size`가 서로 다른) `DATA` 레이어를 가지고 있으며, 하나는 학습 단계, 다른 하나는 검증 단계에 사용됩니다. 또한 `lenet_solver.prototxt`에 정의된 대로, `TEST` 단계에만 있는 `Accuracy` 레이어는 100번의 반복마다 모델의 정확도를 보고하는 역할을 합니다.
(In the above example, this layer will be included only in `TRAIN` phase. If we change `TRAIN` with `TEST`, then this layer will be used only in test phase. By default, that is without layer rules, a layer is always included in the network. Thus, `lenet_train_test.prototxt` has two `DATA` layers defined (with different `batch_size`), one for the training phase and one for the testing phase. Also, there is an `Accuracy` layer which is included only in `TEST` phase for reporting the model accuracy every 100 iteration, as defined in `lenet_solver.prototxt`.)

## MNIST 연산기 만들기 (Define the MNIST Solver)

prototxt 파일 `$CAFFE_ROOT/examples/mnist/lenet_solver.prototxt`의 각 줄을 설명하는 주석을 잘 보세요.
(Check out the comments explaining each line in the prototxt `$CAFFE_ROOT/examples/mnist/lenet_solver.prototxt`:)

    # 학습/검증 신경망 프로토콜 버퍼 정의
    # The train/test net protocol buffer definition
    net: "examples/mnist/lenet_train_test.prototxt"
    # test_iter는 검증할 때 몇 번의 전진 단계를 거칠지를 정합니다.
    # MNIST의 경우 검증 묶음 단위를 100으로 하고 검증을 100번 반복해서,
    # 전체 10000개의 이미지를 포괄합니다.
    # test_iter specifies how many forward passes the test should carry out.
    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
    # covering the full 10,000 testing images.
    test_iter: 100
    # 학습 500번 반복마다 한 번씩 검증을 합니다.
    # Carry out testing every 500 training iterations.
    test_interval: 500
    # 신경망의 기본 학습률, 관성(momentum), 가중치 붕괴를 설정합니다.
    # The base learning rate, momentum and the weight decay of the network.
    base_lr: 0.01
    momentum: 0.9
    weight_decay: 0.0005
    # 학습률 원칙
    # The learning rate policy
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    # 100번의 반복마다 화면에 표시합니다.
    # Display every 100 iterations
    display: 100
    # 최대 반복 회수
    # The maximum number of iterations
    max_iter: 10000
    # 중간 결과 기록
    # snapshot intermediate results
    snapshot: 5000
    snapshot_prefix: "examples/mnist/lenet"
    # 연산기 방식: CPU 혹은 GPU
    # solver mode: CPU or GPU
    solver_mode: GPU

## 모델 학습시키고 검증하기 (Training and Testing the Model)

모델 학습은 신경망 정의와 연산기 프로토버프 파일만 작성하면 간단히 끝납니다. 간단히 `train_lenet.sh`를 실행하거나 다음 명령어를 직접 입력하세요.
(Training the model is simple after you have written the network definition protobuf and solver protobuf files. Simply run `train_lenet.sh`, or the following command directly:)

    cd $CAFFE_ROOT
    ./examples/mnist/train_lenet.sh

`train_lenet.sh` 는 단순한 스크립트이지만 간단히 설명하겠습니다. 학습에 사용되는 도구는 `caffe`이고 연산기 프로토버프 문서 파일을 인자로 받아서 `train` 작업을 합니다.
(`train_lenet.sh` is a simple script, but here is a quick explanation: the main tool for training is `caffe` with action `train` and the solver protobuf text file as its argument.)

위의 코드를 실행하면 아래와 같이 엄청난 알림이 날라다니는 것을 볼 수 있습니다 ^^
(When you run the code, you will see a lot of messages flying by like this:))

    I1203 net.cpp:66] Creating Layer conv1
    I1203 net.cpp:76] conv1 <- data
    I1203 net.cpp:101] conv1 -> conv1
    I1203 net.cpp:116] Top shape: 20 24 24
    I1203 net.cpp:127] conv1 needs backward computation.

이 알림들은 각 레이어의 세부 사항, 연결, 출력의 모양에 대한 것이며 디버깅할 때에 유용할 수 있습니다. 초기화 후에 학습이 시작됩니다.
(These messages tell you the details about each layer, its connections and its output shape, which may be helpful in debugging. After the initialization, the training will start:)

    I1203 net.cpp:142] Network initialization done.
    I1203 solver.cpp:36] Solver scaffolding done.
    I1203 solver.cpp:44] Solving LeNet

연산기 설정에 따라 학습 손실 함수를 100회 반복시마다 출력하고 신경망을 500회 반복시마다 검증합니다. 이런 알림을 보게 됩니다 ^^
(Based on the solver setting, we will print the training loss function every 100 iterations, and test the network every 500 iterations. You will see messages like this:))

    I1203 solver.cpp:204] Iteration 100, lr = 0.00992565
    I1203 solver.cpp:66] Iteration 100, loss = 0.26044
    ...
    I1203 solver.cpp:84] Testing net
    I1203 solver.cpp:111] Test score #0: 0.9785
    I1203 solver.cpp:111] Test score #1: 0.0606671

학습 반복 매 회에 대해 `lr`은 그 반복에서의 학습률이고 `loss`는 학습 함수입니다. (번역자 의견: 학습 함수가 아니라 학습 손실 함수인데 잘못 쓴 것 같습니다.) 검증 단계의 출력의 경우 score 0은 정확도이고 score 1은  검증 손실 함수입니다.
(For each training iteration, `lr` is the learning rate of that iteration, and `loss` is the training function. For the output of the testing phase, score 0 is the accuracy, and score 1 is the testing loss function.)

그리고 몇 분 후면, 끝납니다!
(And after a few minutes, you are done!)

    I1203 solver.cpp:84] Testing net
    I1203 solver.cpp:111] Test score #0: 0.9897
    I1203 solver.cpp:111] Test score #1: 0.0324599
    I1203 solver.cpp:126] Snapshotting to lenet_iter_10000
    I1203 solver.cpp:133] Snapshotting solver state to lenet_iter_10000.solverstate
    I1203 solver.cpp:78] Optimization Done.

이진 프로토버프 파일로 저장되는 최종 모델의 이름은
(The final model, stored as a binary protobuf file, is stored at)

    lenet_iter_10000

이며, 현실에 적용 가능한 데이터셋으로 학습된 경우 학습된 모델로써 당신의 응용 프로그램에서 사용될 수 있습니다.
(which you can deploy as a trained model in your application, if you are training on a real-world application dataset.)

### 어... GPU를 사용한 학습은요? (Um... How about GPU training?)

이미 여러분이 했습니다! 모든 학습은 GPU에서 돌아갑니다. 사실, 만약 CPU에서 학습을 시키고 싶으면 `lenet_solver.prototxt`에서 그저 한 줄만 바꾸면 됩니다.
(You just did! All the training was carried out on the GPU. In fact, if you would like to do training on CPU, you can simply change one line in `lenet_solver.prototxt`:)

    # 연산기 방식: CPU 혹은 GPU
    # solver mode: CPU or GPU
    solver_mode: CPU

그러면 학습에 CPU를 사용합니다. 참 쉽지요?
(and you will be using CPU for training. Isn’t that easy?)

MNIST는 작은 데이터셋이기 때문에, 통신 부하 때문에 GPU에서 학습시키는 것이 그렇게 큰 장점이 되지 않습니다. 이미지넷(ImageNet)과 같이 더 큰 데이터셋과 더 복잡한 모델을 사용할 경우 계산 속도 차이가 더욱 확실할 것입니다.
(MNIST is a small dataset, so training with GPU does not really introduce too much benefit due to communication overheads. On larger datasets with more complex models, such as ImageNet, the computation speed difference will be more significant.)

### 어떻게 하면 고정된 반복마다 학습률을 감소시킬 수 있나요? (How to reduce the learning rate at fixed steps?)

`lenet_multistep_solver.prototxt`를 보세요.
(Look at lenet_multistep_solver.prototxt)
