# 연산기 (Solver)

연산기는 신경망의 전진 과정과 후진 벡터 기울기를 조정을 통해 인자 갱신(update)값을 얻어내어 손실을 향상시키는 시도를 함을 통해 모델 최적화를 총괄합니다. 학습에 대한 책임은 최적화를 감독하며 인자 갱신값을 생성하는 연산기와 손실과 기울기를 만들어내는 신경망에게 분배되어 있습니다.
(The solver orchestrates model optimization by coordinating the network’s forward inference and backward gradients to form parameter updates that attempt to improve the loss. The responsibilities of learning are divided between the Solver for overseeing the optimization and generating parameter updates and the Net for yielding loss and gradients.)

카페 연산기들은 다음과 같습니다.
(The Caffe solvers are:)

* 확률적 경사 하강법 (`type: "SGD"`)
(Stochastic Gradient Descent (`type: "SGD"`),)
* 조정 가능한 학습률 방식 (AdaDelta) (`type: "AdaDelta"`)
(AdaDelta (`type: "AdaDelta"`),)
* 조정 가능한 기울기 (`type: "AdaGrad"`)
(Adaptive Gradient (`type: "AdaGrad"`),)
* 아담 (`type: "Adam"`)
(Adam (`type: "Adam"`),)
* 네스테로프 가속 기울기 (`type: "Nesterov"`)
Nesterov’s Accelerated Gradient (`type: "Nesterov"`) and
* 기울기를 최근 크기의 평균으로 나누기 (RMSprop) (`type: "RMSProp"`)
RMSprop (`type: "RMSProp"`)

연산기는 다음과 같은 일을 합니다.
(The solver)

1. 최적화 기록에 대한 뼈대를 세우고 학습을 위한 학습 신경망과 평가를 위한 테스트 신경망을 만듭니다.
(scaffolds the optimization bookkeeping and creates the training network for learning and test network(s) for evaluation.)
2. 전진과 후진을 호출하고 인자를 갱신하는 과정을 반복함으로써 최적화를 합니다.
(iteratively optimizes by calling forward / backward and updating parameters)
3. (주기적으로) 테스트 신경망을 평가합니다.
((periodically) evaluates the test networks)
4. 최적화 과정에서의 모델과 연산기 상태를 기록합니다.
(snapshots the model and solver state throughout the optimization)

그리고 각각의 반복마다
(where each iteration)

1. 신경망을 전진시켜 출력과 손실을 계산합니다.
(calls network forward to compute the output and loss)
2. 신경망을 후진시켜 기울기를 계산합니다.
(calls network backward to compute the gradients)
3. 기울기를 연산기 방식에 맞추어 인자 갱신에 사용합니다.
(incorporates the gradients into parameter updates according to the solver method)
4. 연산기를 학습률, 과거, 방식에 맞추어 갱신합니다.
(updates the solver state according to learning rate, history, and method)

위의 과정을 통해 처음부터 학습된 모델까지의 모든 과정에 걸친 가중치를 얻어냅니다.
(to take the weights all the way from initialization to learned model.)

카페 모델의 경우와 같이, 카페 연산기도 CPU / GPU 상태에서 실행됩니다.
(Like Caffe models, Caffe solvers run in CPU / GPU modes.)

## 방식 (Methods)

연산기의 방식은 손실 최소화에 대한 일반적인 최적화 문제를 다룹니다. D라는 데이터셋에 대한 최적화 대상은 데이터셋 전체에 걸친 모든 데이터 항목 |D|개에 대한 평균 손실인 L(W) 입니다.
(The solver methods address the general optimization problem of loss minimization. For dataset D, the optimization objective is the average loss over all |D| data instances throughout the dataset)

L(W)=(1/|D|)∑\_i^|D|^f\_W(X^(i))+λr(W)

여기서 f_W(X^(i))는 데이터 항목 X^(i)로 인한 손실이고 r(W)는 가중치 λ가 사용된 정형화입니다. |D|는 매우 클 수 있으며, 따라서 현업에서는 목표를 달성하기 위해 각각의 연산기 반복에 대해 확률적 근사를 사용하여 N<<|D|개의 항목으로 구성된 미니배치를 만듭니다.
(where f_W(X^(i)) is the loss on data instance X^(i) and r(W) is a regularization term with weight λ. |D| can be very large, so in practice, in each solver iteration we use a stochastic approximation of this objective, drawing a mini-batch of N<<|D| instances:)

L(W)≈(1/N)∑\_i^Nf\_W(X^(i))+λr(W)

모델은 전진 과정에서 f\_W를 계산하고 후진 과정에서 기울기인 ∇f\_W를 계산합니다.
(The model computes f\_W in the forward pass and the gradient ∇f\_W in the backward pass.)

인자 갱신값인 ΔW는 에러 기울기인 ∇f\_W, 정형화 기울기인 ∇r(W)와 그 외 각 방식에 따른 요소들을 사용해서 연산기가 계산해냅니다.
(The parameter update ΔW is formed by the solver from the error gradient ∇f\_W, the regularization gradient ∇r(W), and other particulars to each method.)

### 확률적 경사 하강법 (SGD)

Stochastic gradient descent (type: "SGD") updates the weights WW by a linear combination of the negative gradient ∇L(W)∇L(W) and the previous weight update VtVt. The learning rate αα is the weight of the negative gradient. The momentum μμ is the weight of the previous update.

Formally, we have the following formulas to compute the update value Vt+1Vt+1 and the updated weights Wt+1Wt+1 at iteration t+1t+1, given the previous weight update VtVt and current weights WtWt:

Vt+1=μVt−α∇L(Wt)
Vt+1=μVt−α∇L(Wt)
Wt+1=Wt+Vt+1
Wt+1=Wt+Vt+1
The learning “hyperparameters” (αα and μμ) might require a bit of tuning for best results. If you’re not sure where to start, take a look at the “Rules of thumb” below, and for further information you might refer to Leon Bottou’s Stochastic Gradient Descent Tricks [1].

[1] L. Bottou. Stochastic Gradient Descent Tricks. Neural Networks: Tricks of the Trade: Springer, 2012.

Rules of thumb for setting the learning rate αα and momentum μμ
A good strategy for deep learning with SGD is to initialize the learning rate αα to a value around α≈0.01=10−2α≈0.01=10−2, and dropping it by a constant factor (e.g., 10) throughout training when the loss begins to reach an apparent “plateau”, repeating this several times. Generally, you probably want to use a momentum μ=0.9μ=0.9 or similar value. By smoothing the weight updates across iterations, momentum tends to make deep learning with SGD both stabler and faster.

This was the strategy used by Krizhevsky et al. [1] in their famously winning CNN entry to the ILSVRC-2012 competition, and Caffe makes this strategy easy to implement in a SolverParameter, as in our reproduction of [1] at ./examples/imagenet/alexnet_solver.prototxt.

To use a learning rate policy like this, you can put the following lines somewhere in your solver prototxt file:

    base_lr: 0.01     # begin training at a learning rate of 0.01 = 1e-2
    
    lr_policy: "step" # learning rate policy: drop the learning rate in "steps"
                      # by a factor of gamma every stepsize iterations
    
    gamma: 0.1        # drop the learning rate by a factor of 10
                      # (i.e., multiply it by a factor of gamma = 0.1)
    
    stepsize: 100000  # drop the learning rate every 100K iterations
    
    max_iter: 350000  # train for 350K iterations total
    
    momentum: 0.9
Under the above settings, we’ll always use momentum μ=0.9μ=0.9. We’ll begin training at a base_lr of α=0.01=10−2α=0.01=10−2 for the first 100,000 iterations, then multiply the learning rate by gamma (γγ) and train at α′=αγ=(0.01)(0.1)=0.001=10−3α′=αγ=(0.01)(0.1)=0.001=10−3 for iterations 100K-200K, then at α′′=10−4α″=10−4 for iterations 200K-300K, and finally train until iteration 350K (since we have max_iter: 350000) at α′′′=10−5α‴=10−5.

Note that the momentum setting μμ effectively multiplies the size of your updates by a factor of 11−μ11−μ after many iterations of training, so if you increase μμ, it may be a good idea to decrease αα accordingly (and vice versa).

For example, with μ=0.9μ=0.9, we have an effective update size multiplier of 11−0.9=1011−0.9=10. If we increased the momentum to μ=0.99μ=0.99, we’ve increased our update size multiplier to 100, so we should drop αα (base_lr) by a factor of 10.

Note also that the above settings are merely guidelines, and they’re definitely not guaranteed to be optimal (or even work at all!) in every situation. If learning diverges (e.g., you start to see very large or NaN or inf loss values or outputs), try dropping the base_lr (e.g., base_lr: 0.001) and re-training, repeating this until you find a base_lr value that works.

[1] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems, 2012.

### 조정 가능한 학습률 방식 (AdaDelta)

The AdaDelta (type: "AdaDelta") method (M. Zeiler [1]) is a “robust learning rate method”. It is a gradient-based optimization method (like SGD). The update formulas are

(vt)iRMS(∇L(Wt))iE[g2]t=RMS((vt−1)i)RMS(∇L(Wt))i(∇L(Wt′))i=E[g2]+ε−−−−−−−−√=δE[g2]t−1+(1−δ)g2t
(vt)i=RMS⁡((vt−1)i)RMS⁡(∇L(Wt))i(∇L(Wt′))iRMS⁡(∇L(Wt))i=E[g2]+εE[g2]t=δE[g2]t−1+(1−δ)gt2
and

(Wt+1)i=(Wt)i−α(vt)i.
(Wt+1)i=(Wt)i−α(vt)i.
[1] M. Zeiler ADADELTA: AN ADAPTIVE LEARNING RATE METHOD. arXiv preprint, 2012.

### 조정 가능한 기울기 (AdaGrad)

The adaptive gradient (type: "AdaGrad") method (Duchi et al. [1]) is a gradient-based optimization method (like SGD) that attempts to “find needles in haystacks in the form of very predictive but rarely seen features,” in Duchi et al.’s words. Given the update information from all previous iterations (∇L(W))t′(∇L(W))t′ for t′∈{1,2,...,t}t′∈{1,2,...,t}, the update formulas proposed by [1] are as follows, specified for each component ii of the weights WW:

(Wt+1)i=(Wt)i−α(∇L(Wt))i∑tt′=1(∇L(Wt′))2i−−−−−−−−−−−−−−√
(Wt+1)i=(Wt)i−α(∇L(Wt))i∑t′=1t(∇L(Wt′))i2
Note that in practice, for weights W∈RdW∈Rd, AdaGrad implementations (including the one in Caffe) use only O(d)O(d) extra storage for the historical gradient information (rather than the O(dt)O(dt) storage that would be necessary to store each historical gradient individually).

[1] J. Duchi, E. Hazan, and Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. The Journal of Machine Learning Research, 2011.

## 아담 (Adam)

The Adam (type: "Adam"), proposed in Kingma et al. [1], is a gradient-based optimization method (like SGD). This includes an “adaptive moment estimation” (mt,vtmt,vt) and can be regarded as a generalization of AdaGrad. The update formulas are

(mt)i=β1(mt−1)i+(1−β1)(∇L(Wt))i,(vt)i=β2(vt−1)i+(1−β2)(∇L(Wt))2i
(mt)i=β1(mt−1)i+(1−β1)(∇L(Wt))i,(vt)i=β2(vt−1)i+(1−β2)(∇L(Wt))i2
and

(Wt+1)i=(Wt)i−α1−(β2)ti−−−−−−−√1−(β1)ti(mt)i(vt)i−−−−√+ε.
(Wt+1)i=(Wt)i−α1−(β2)it1−(β1)it(mt)i(vt)i+ε.
Kingma et al. [1] proposed to use β1=0.9,β2=0.999,ε=10−8β1=0.9,β2=0.999,ε=10−8 as default values. Caffe uses the values of momemtum, momentum2, delta for β1,β2,εβ1,β2,ε, respectively.

[1] D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. International Conference for Learning Representations, 2015.

### 네스테로프 가속 기울기 (NAG)

Nesterov’s accelerated gradient (type: "Nesterov") was proposed by Nesterov [1] as an “optimal” method of convex optimization, achieving a convergence rate of O(1/t2)O(1/t2) rather than the O(1/t)O(1/t). Though the required assumptions to achieve the O(1/t2)O(1/t2) convergence typically will not hold for deep networks trained with Caffe (e.g., due to non-smoothness and non-convexity), in practice NAG can be a very effective method for optimizing certain types of deep learning architectures, as demonstrated for deep MNIST autoencoders by Sutskever et al. [2].

The weight update formulas look very similar to the SGD updates given above:

Vt+1=μVt−α∇L(Wt+μVt)
Vt+1=μVt−α∇L(Wt+μVt)
Wt+1=Wt+Vt+1
Wt+1=Wt+Vt+1
What distinguishes the method from SGD is the weight setting WW on which we compute the error gradient ∇L(W)∇L(W) – in NAG we take the gradient on weights with added momentum ∇L(Wt+μVt)∇L(Wt+μVt); in SGD we simply take the gradient ∇L(Wt)∇L(Wt) on the current weights themselves.

[1] Y. Nesterov. A Method of Solving a Convex Programming Problem with Convergence Rate O(1/k√)O(1/k). Soviet Mathematics Doklady, 1983.

[2] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the Importance of Initialization and Momentum in Deep Learning. Proceedings of the 30th International Conference on Machine Learning, 2013.

### 기울기를 최근 크기의 평균으로 나누기 (RMSprop)

The RMSprop (type: "RMSProp"), suggested by Tieleman in a Coursera course lecture, is a gradient-based optimization method (like SGD). The update formulas are

(vt)i={(vt−1)i+δ,(vt−1)i⋅(1−δ),(∇L(Wt))i(∇L(Wt−1))i>0else
(vt)i={(vt−1)i+δ,(∇L(Wt))i(∇L(Wt−1))i>0(vt−1)i⋅(1−δ),else
(Wt+1)i=(Wt)i−α(vt)i,
(Wt+1)i=(Wt)i−α(vt)i,
If the gradient updates results in oscillations the gradient is reduced by times 1−δ1−δ. Otherwise it will be increased by δδ. The default value of δδ (rms_decay) is set to δ=0.02δ=0.02.

[1] T. Tieleman, and G. Hinton. RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning.Technical report, 2012.

## 뼈대 세우기 (Scaffolding)

The solver scaffolding prepares the optimization method and initializes the model to be learned in `Solver::Presolve()`.

    > caffe train -solver examples/mnist/lenet_solver.prototxt
    I0902 13:35:56.474978 16020 caffe.cpp:90] Starting Optimization
    I0902 13:35:56.475190 16020 solver.cpp:32] Initializing solver from parameters:
    test_iter: 100
    test_interval: 500
    base_lr: 0.01
    display: 100
    max_iter: 10000
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    momentum: 0.9
    weight_decay: 0.0005
    snapshot: 5000
    snapshot_prefix: "examples/mnist/lenet"
    solver_mode: GPU
    net: "examples/mnist/lenet_train_test.prototxt"
Net initialization

    I0902 13:35:56.655681 16020 solver.cpp:72] Creating training net from net file: examples/mnist/lenet_train_test.prototxt
    [...]
    I0902 13:35:56.656740 16020 net.cpp:56] Memory required for data: 0
    I0902 13:35:56.656791 16020 net.cpp:67] Creating Layer mnist
    I0902 13:35:56.656811 16020 net.cpp:356] mnist -> data
    I0902 13:35:56.656846 16020 net.cpp:356] mnist -> label
    I0902 13:35:56.656874 16020 net.cpp:96] Setting up mnist
    I0902 13:35:56.694052 16020 data_layer.cpp:135] Opening lmdb examples/mnist/mnist_train_lmdb
    I0902 13:35:56.701062 16020 data_layer.cpp:195] output data size: 64,1,28,28
    I0902 13:35:56.701146 16020 data_layer.cpp:236] Initializing prefetch
    I0902 13:35:56.701196 16020 data_layer.cpp:238] Prefetch initialized.
    I0902 13:35:56.701212 16020 net.cpp:103] Top shape: 64 1 28 28 (50176)
    I0902 13:35:56.701230 16020 net.cpp:103] Top shape: 64 1 1 1 (64)
    [...]
    I0902 13:35:56.703737 16020 net.cpp:67] Creating Layer ip1
    I0902 13:35:56.703753 16020 net.cpp:394] ip1 <- pool2
    I0902 13:35:56.703778 16020 net.cpp:356] ip1 -> ip1
    I0902 13:35:56.703797 16020 net.cpp:96] Setting up ip1
    I0902 13:35:56.728127 16020 net.cpp:103] Top shape: 64 500 1 1 (32000)
    I0902 13:35:56.728142 16020 net.cpp:113] Memory required for data: 5039360
    I0902 13:35:56.728175 16020 net.cpp:67] Creating Layer relu1
    I0902 13:35:56.728194 16020 net.cpp:394] relu1 <- ip1
    I0902 13:35:56.728219 16020 net.cpp:345] relu1 -> ip1 (in-place)
    I0902 13:35:56.728240 16020 net.cpp:96] Setting up relu1
    I0902 13:35:56.728256 16020 net.cpp:103] Top shape: 64 500 1 1 (32000)
    I0902 13:35:56.728270 16020 net.cpp:113] Memory required for data: 5167360
    I0902 13:35:56.728287 16020 net.cpp:67] Creating Layer ip2
    I0902 13:35:56.728304 16020 net.cpp:394] ip2 <- ip1
    I0902 13:35:56.728333 16020 net.cpp:356] ip2 -> ip2
    I0902 13:35:56.728356 16020 net.cpp:96] Setting up ip2
    I0902 13:35:56.728690 16020 net.cpp:103] Top shape: 64 10 1 1 (640)
    I0902 13:35:56.728705 16020 net.cpp:113] Memory required for data: 5169920
    I0902 13:35:56.728734 16020 net.cpp:67] Creating Layer loss
    I0902 13:35:56.728747 16020 net.cpp:394] loss <- ip2
    I0902 13:35:56.728767 16020 net.cpp:394] loss <- label
    I0902 13:35:56.728786 16020 net.cpp:356] loss -> loss
    I0902 13:35:56.728811 16020 net.cpp:96] Setting up loss
    I0902 13:35:56.728837 16020 net.cpp:103] Top shape: 1 1 1 1 (1)
    I0902 13:35:56.728849 16020 net.cpp:109]     with loss weight 1
    I0902 13:35:56.728878 16020 net.cpp:113] Memory required for data: 5169924
Loss

    I0902 13:35:56.728893 16020 net.cpp:170] loss needs backward computation.
    I0902 13:35:56.728909 16020 net.cpp:170] ip2 needs backward computation.
    I0902 13:35:56.728924 16020 net.cpp:170] relu1 needs backward computation.
    I0902 13:35:56.728938 16020 net.cpp:170] ip1 needs backward computation.
    I0902 13:35:56.728953 16020 net.cpp:170] pool2 needs backward computation.
    I0902 13:35:56.728970 16020 net.cpp:170] conv2 needs backward computation.
    I0902 13:35:56.728984 16020 net.cpp:170] pool1 needs backward computation.
    I0902 13:35:56.728998 16020 net.cpp:170] conv1 needs backward computation.
    I0902 13:35:56.729014 16020 net.cpp:172] mnist does not need backward computation.
    I0902 13:35:56.729027 16020 net.cpp:208] This network produces output loss
    I0902 13:35:56.729053 16020 net.cpp:467] Collecting Learning Rate and Weight Decay.
    I0902 13:35:56.729071 16020 net.cpp:219] Network initialization done.
    I0902 13:35:56.729085 16020 net.cpp:220] Memory required for data: 5169924
    I0902 13:35:56.729277 16020 solver.cpp:156] Creating test net (#0) specified by net file: examples/mnist/lenet_train_test.prototxt
Completion

    I0902 13:35:56.806970 16020 solver.cpp:46] Solver scaffolding done.
    I0902 13:35:56.806984 16020 solver.cpp:165] Solving LeNet

## 인자 갱신 (Updating Parameters)

The actual weight update is made by the solver then applied to the net parameters in Solver::ComputeUpdateValue(). The ComputeUpdateValue method incorporates any weight decay r(W)r(W) into the weight gradients (which currently just contain the error gradients) to get the final gradient with respect to each network weight. Then these gradients are scaled by the learning rate αα and the update to subtract is stored in each parameter Blob’s diff field. Finally, the Blob::Update method is called on each parameter blob, which performs the final update (subtracting the Blob’s diff from its data).

## 저장하고 재개하기 (Snapshotting and Resuming)

The solver snapshots the weights and its own state during training in Solver::Snapshot() and Solver::SnapshotSolverState(). The weight snapshots export the learned model while the solver snapshots allow training to be resumed from a given point. Training is resumed by Solver::Restore() and Solver::RestoreSolverState().

Weights are saved without extension while solver states are saved with .solverstate extension. Both files will have an _iter_N suffix for the snapshot iteration number.

Snapshotting is configured by:

    # The snapshot interval in iterations.
    snapshot: 5000
    # File path prefix for snapshotting model weights and solver state.
    # Note: this is relative to the invocation of the `caffe` utility, not the
    # solver definition file.
    snapshot_prefix: "/path/to/model"
    # Snapshot the diff along with the weights. This can help debugging training
    # but takes more storage.
    snapshot_diff: false
    # A final snapshot is saved at the end of training unless
    # this flag is set to false. The default is true.
    snapshot_after_train: true
in the solver definition prototxt.
