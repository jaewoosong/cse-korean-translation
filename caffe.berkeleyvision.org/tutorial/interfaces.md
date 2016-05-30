# 인터페이스 (Interfaces)

카페는 명령줄, 파이썬, 매트랩 인터페이스를 일상 용도, 연구 코드와의 교류, 신속한 시험용 원형 제작을 위해 제공합니다. 카페의 핵심은 C++ 라이브러리이고 개발을 위한 모듈형 인터페이스를 제공하지만, 모든 경우에 항상 맞춤형 컴파일이 필요한 것은 아닙니다. 여러분을 위해 cmdcaffe, pycaffe, matcaffe 인터페이스가 준비되어 있습니다.
(Caffe has command line, Python, and MATLAB interfaces for day-to-day usage, interfacing with research code, and rapid prototyping. While Caffe is a C++ library at heart and it exposes a modular interface for development, not every occasion calls for custom compilation. The cmdcaffe, pycaffe, and matcaffe interfaces are here for you.)

## 명령줄 (Command Line)

명령줄 인터페이스 -- cmdcaffe -- 는 모델 학습, 평가, 진단을 위한 `caffe` 도구입니다. 도움말을 보려면 아무런 인자 없이 `caffe`를 실행하세요. 이것 및 다른 도구는 caffe/build/tools에서 찾을 수 있습니다. (이하 예제는 Lenet / MNIST 예제를 먼저 끝냈어야 동작합니다.)
(The command line interface -- cmdcaffe -- is the `caffe` tool for model training, scoring, and diagnostics. Run `caffe` without any arguments for help. This tool and others are found in caffe/build/tools. (The following example calls require completing the LeNet / MNIST example first.))

**학습**: `caffe train`은 모델을 맨 처음부터 학습할 때, 저장된 중간 단계 모델로부터 학습을 재개할 때, 새로운 데이터와 작업에 대해 모델을 세부 조정할 때 사용됩니다.
(**Training**: `caffe train` learns models from scratch, resumes learning from saved snapshots, and fine-tunes models to new data and tasks:)

* 모든 학습은 `-solver solver.prototxt` 인자를 통한 연산기 설정을 필요로 합니다.
(All training requires a solver configuration through the `-solver solver.prototxt` argument.)
* 학습 재개를 하려면 `-snapshot model_iter_1000.solverstate` 인자를 통해 연산기 중간 단계를 불러와야 합니다.
(Resuming requires the `-snapshot model_iter_1000.solverstate` argument to load the solver snapshot.)
* 모델 세부 조정에는 모델 초기화를 위해 `-weights model.caffemodel` 인자가 필요합니다.
(Fine-tuning requires the `-weights model.caffemodel` argument for the model initialization.)

예를 들어, 다음과 같이 실행할 수 있습니다.
(For example, you can run:)

    # LeNet 학습
    # train LeNet
    caffe train -solver examples/mnist/lenet_solver.prototxt
    # 2번 GPU에서 학습
    # train on GPU 2
    caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2
    # 중간 단계 모델에서 학습 재개
    # resume training from the half-way point snapshot
    caffe train -solver examples/mnist/lenet_solver.prototxt -snapshot examples/mnist/lenet_iter_5000.solverstate

세부 조정에 대한 전체 예제는 examples/finetuning\_on\_flickr\_style에 있고, 학습 실행 자체는 다음과 같습니다.
(For a full example of fine-tuning, see examples/finetuning\_on\_flickr\_style, but the training call alone is)

    # 스타일 인식을 위한 CaffeNet 모델 가중치 세부 조정
    # fine-tune CaffeNet model weights for style recognition
    caffe train -solver examples/finetuning_on_flickr_style/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

**검사**: `caffe test`는 모델을 검사 단계에서 실행한 뒤 신경망의 출력을 점수로 기록합니다. 신경망은 정확성 혹은 손실 값을 출력하도록 제대로 설계되어야 합니다. 한 번의 일괄 처리마다의 점수가 보고되고, 최종 평균이 마지막에 보고됩니다.
(**Testing**: `caffe test` scores models by running them in the test phase and reports the net output as its score. The net architecture must be properly defined to output an accuracy measure or loss as its output. The per-batch score is reported and then the grand average is reported last.)

    # lenet_train_test.prototxt에 기록된 모델 설계대로
    # 학습된 LeNet 모델을 확인(validation) 데이터에 대해 채점합니다.
    # score the learned LeNet model on the validation set as defined in the
    # model architeture lenet_train_test.prototxt
    caffe test -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 100

**성능 비교**: `caffe time`은 각 레이어에 대한 모델 실행 성능을 시간과 동기화에 대해 비교합니다. 기기의 성능과 모델 별 상대적 실행 시간을 측정할 때에 유용합니다.
(**Benchmarking**: `caffe time` benchmarks model execution layer-by-layer through timing and synchronization. This is useful to check system performance and measure relative execution times for models.)

    # (이 예제를 실행하기 위해서는 LeNet / MNIST 예제를 먼저 끝내야 합니다.)
    # (These example calls require you complete the LeNet / MNIST example first.)
    # LeNet을 CPU를 사용하여 10회 반복 학습하는 데에 걸리는 시간
    # time LeNet training on CPU for 10 iterations
    caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10
    # LeNet을 GPU를 사용하여 기본인 50회 반복 학습하는 데에 걸리는 시간
    # time LeNet training on GPU for the default 50 iterations
    caffe time -model examples/mnist/lenet_train_test.prototxt -gpu 0
    # 주어진 모델 설정과 가중치로 첫 번째 GPU를 사용하여 10회 반복 학습하는 데에 걸리는 시간
    # time a model architecture with the given weights on the first GPU for 10 iterations
    caffe time -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 10

**진단**: `caffe device_query`는 참조를 위해 GPU 세부 사항을 보고하며 GPU가 여러 대 있을 경우 특정 기기를 사용하기 위해 기기의 번호를 검사합니다.
(**Diagnostics**: `caffe device_query` reports GPU details for reference and checking device ordinals for running on a given device in multi-GPU machines.)

    # 첫 번째 기기 접촉
    # query the first device
    caffe device_query -gpu 0

**병렬화**: `caffe` 도구의 `-gpu` 신호는 여러 개의 GPU를 사용하기 위해 쉼표로 분리된 명단을 받을 수 있습니다. 연산기와 신경망이 각각의 GPU에 대해 실행되고, 일괄 처리 크기는 GPU 개수에 대해 효과적으로 곱해집니다. 단일 GPU 학습을 재현하고 싶으면 신경망 정의의 일괄 처리 크기 부분을 알맞게 줄이세요.
(**Parallelism**: the `-gpu` flag to the `caffe` tool can take a comma separated list of IDs to run on multiple GPUs. A solver and net will be instantiated for each GPU so the batch size is effectively multiplied by the number of GPUs. To reproduce single GPU training, reduce the batch size in the network definition accordingly.)

    # GPU 0번과 1번에서 학습 (일괄 처리 크기 두 배로 늘림)
    # train on GPUs 0 & 1 (doubling the batch size)
    caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 0,1
    # 모든 GPU에서 학습 (일괄 처리 크기레 기기 개수를 곱함)
    # train on all GPUs (multiplying batch size by number of devices)
    caffe train -solver examples/mnist/lenet_solver.prototxt -gpu all

## 파이썬 (Python)

The Python interface -- pycaffe -- is the `caffe` module and its scripts in caffe/python. `import caffe` to load models, do forward and backward, handle IO, visualize networks, and even instrument model solving. All model data, derivatives, and parameters are exposed for reading and writing.

- `caffe.Net` is the central interface for loading, configuring, and running models. `caffe.Classifier` and `caffe.Detector` provide convenience interfaces for common tasks.
- `caffe.SGDSolver` exposes the solving interface.
- `caffe.io` handles input / output with preprocessing and protocol buffers.
- `caffe.draw` visualizes network architectures.
- Caffe blobs are exposed as numpy ndarrays for ease-of-use and efficiency.

Tutorial IPython notebooks are found in caffe/examples: do `ipython notebook caffe/examples` to try them. For developer reference docstrings can be found throughout the code.

Compile pycaffe by `make pycaffe`.
Add the module directory to your `$PYTHONPATH` by `export PYTHONPATH=/path/to/caffe/python:$PYTHONPATH` or the like for `import caffe`.

## MATLAB

The MATLAB interface -- matcaffe -- is the `caffe` package in caffe/matlab in which you can integrate Caffe in your Matlab code.

In MatCaffe, you can

* Creating multiple Nets in Matlab
* Do forward and backward computation
* Access any layer within a network, and any parameter blob in a layer
* Get and set data or diff to any blob within a network, not restricting to input blobs or output blobs
* Save a network's parameters to file, and load parameters from file
* Reshape a blob and reshape a network
* Edit network parameter and do network surgery
* Create multiple Solvers in Matlab for training
* Resume training from solver snapshots
* Access train net and test nets in a solver
* Run for a certain number of iterations and give back control to Matlab
* Intermingle arbitrary Matlab code with gradient steps

An ILSVRC image classification demo is in caffe/matlab/demo/classification_demo.m (you need to download BVLC CaffeNet from [Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) to run it).

### Build MatCaffe

Build MatCaffe with `make all matcaffe`. After that, you may test it using `make mattest`.

Common issue: if you run into error messages like `libstdc++.so.6:version 'GLIBCXX_3.4.15' not found` during `make mattest`, then it usually means that your Matlab's runtime libraries do not match your compile-time libraries. You may need to do the following before you start Matlab:

    export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

Or the equivalent based on where things are installed on your system, and do `make mattest` again to see if the issue is fixed. Note: this issue is sometimes more complicated since during its startup Matlab may overwrite your `LD_LIBRARY_PATH` environment variable. You can run `!ldd ./matlab/+caffe/private/caffe_.mexa64` (the mex extension may differ on your system) in Matlab to see its runtime libraries, and preload your compile-time libraries by exporting them to your `LD_PRELOAD` environment variable.

After successful building and testing, add this package to Matlab search PATH by starting `matlab` from caffe root folder and running the following commands in Matlab command window.

    addpath ./matlab

You can save your Matlab search PATH by running `savepath` so that you don't have to run the command above again every time you use MatCaffe.

### Use MatCaffe

MatCaffe is very similar to PyCaffe in usage.

Examples below shows detailed usages and assumes you have downloaded BVLC CaffeNet from [Model Zoo](http://caffe.berkeleyvision.org/model_zoo.html) and started `matlab` from caffe root folder.

    model = './models/bvlc_reference_caffenet/deploy.prototxt';
    weights = './models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';

#### Set mode and device

**Mode and device should always be set BEFORE you create a net or a solver.**

Use CPU:

    caffe.set_mode_cpu();

Use GPU and specify its gpu_id:

    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);

#### Create a network and access its layers and blobs

Create a network:

    net = caffe.Net(model, weights, 'test'); % create net and load weights

Or

    net = caffe.Net(model, 'test'); % create net but not load weights
    net.copy_from(weights); % load weights

which creates `net` object as

      Net with properties:

               layer_vec: [1x23 caffe.Layer]
                blob_vec: [1x15 caffe.Blob]
                  inputs: {'data'}
                 outputs: {'prob'}
        name2layer_index: [23x1 containers.Map]
         name2blob_index: [15x1 containers.Map]
             layer_names: {23x1 cell}
              blob_names: {15x1 cell}

The two `containers.Map` objects are useful to find the index of a layer or a blob by its name.

You have access to every blob in this network. To fill blob 'data' with all ones:

    net.blobs('data').set_data(ones(net.blobs('data').shape));

To multiply all values in blob 'data' by 10:

    net.blobs('data').set_data(net.blobs('data').get_data() * 10);

**Be aware that since Matlab is 1-indexed and column-major, the usual 4 blob dimensions in Matlab are `[width, height, channels, num]`, and `width` is the fastest dimension. Also be aware that images are in BGR channels.** Also, Caffe uses single-precision float data. If your data is not single, `set_data` will automatically convert it to single.

You also have access to every layer, so you can do network surgery. For example, to multiply conv1 parameters by 10:

    net.params('conv1', 1).set_data(net.params('conv1', 1).get_data() * 10); % set weights
    net.params('conv1', 2).set_data(net.params('conv1', 2).get_data() * 10); % set bias

Alternatively, you can use

    net.layers('conv1').params(1).set_data(net.layers('conv1').params(1).get_data() * 10);
    net.layers('conv1').params(2).set_data(net.layers('conv1').params(2).get_data() * 10);

To save the network you just modified:

    net.save('my_net.caffemodel');

To get a layer's type (string):

    layer_type = net.layers('conv1').type;

#### Forward and backward

Forward pass can be done using `net.forward` or `net.forward_prefilled`. Function `net.forward` takes in a cell array of N-D arrays containing data of input blob(s) and outputs a cell array containing data from output blob(s). Function `net.forward_prefilled` uses existing data in input blob(s) during forward pass, takes no input and produces no output. After creating some data for input blobs like `data = rand(net.blobs('data').shape);` you can run

    res = net.forward({data});
    prob = res{1};

Or

    net.blobs('data').set_data(data);
    net.forward_prefilled();
    prob = net.blobs('prob').get_data();

Backward is similar using `net.backward` or `net.backward_prefilled` and replacing `get_data` and `set_data` with `get_diff` and `set_diff`. After creating some gradients for output blobs like `prob_diff = rand(net.blobs('prob').shape);` you can run

    res = net.backward({prob_diff});
    data_diff = res{1};

Or

    net.blobs('prob').set_diff(prob_diff);
    net.backward_prefilled();
    data_diff = net.blobs('data').get_diff();

**However, the backward computation above doesn't get correct results, because Caffe decides that the network does not need backward computation. To get correct backward results, you need to set `'force_backward: true'` in your network prototxt.**

After performing forward or backward pass, you can also get the data or diff in internal blobs. For example, to extract pool5 features after forward pass:

    pool5_feat = net.blobs('pool5').get_data();

#### Reshape

Assume you want to run 1 image at a time instead of 10:

    net.blobs('data').reshape([227 227 3 1]); % reshape blob 'data'
    net.reshape();

Then the whole network is reshaped, and now `net.blobs('prob').shape` should be `[1000 1]`;

#### Training

Assume you have created training and validation lmdbs following our [ImageNET Tutorial](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html), to create a solver and train on ILSVRC 2012 classification dataset:

    solver = caffe.Solver('./models/bvlc_reference_caffenet/solver.prototxt');

which creates `solver` object as

      Solver with properties:

              net: [1x1 caffe.Net]
        test_nets: [1x1 caffe.Net]

To train:

    solver.solve();

Or train for only 1000 iterations (so that you can do something to its net before training more iterations)

    solver.step(1000);

To get iteration number:

    iter = solver.iter();

To get its network:

    train_net = solver.net;
    test_net = solver.test_nets(1);

To resume from a snapshot "your_snapshot.solverstate":

    solver.restore('your_snapshot.solverstate');

#### Input and output

`caffe.io` class provides basic input functions `load_image` and `read_mean`. For example, to read ILSVRC 2012 mean file (assume you have downloaded imagenet example auxiliary files by running `./data/ilsvrc12/get_ilsvrc_aux.sh`):

    mean_data = caffe.io.read_mean('./data/ilsvrc12/imagenet_mean.binaryproto');

To read Caffe's example image and resize to `[width, height]` and suppose we want `width = 256; height = 256;`

    im_data = caffe.io.load_image('./examples/images/cat.jpg');
    im_data = imresize(im_data, [width, height]); % resize using Matlab's imresize

**Keep in mind that `width` is the fastest dimension and channels are BGR, which is different from the usual way that Matlab stores an image.** If you don't want to use `caffe.io.load_image` and prefer to load an image by yourself, you can do

    im_data = imread('./examples/images/cat.jpg'); % read image
    im_data = im_data(:, :, [3, 2, 1]); % convert from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]); % permute width and height
    im_data = single(im_data); % convert to single precision

Also, you may take a look at caffe/matlab/demo/classification_demo.m to see how to prepare input by taking crops from an image.

We show in caffe/matlab/hdf5creation how to read and write HDF5 data with Matlab. We do not provide extra functions for data output as Matlab itself is already quite powerful in output.

#### Clear nets and solvers

Call `caffe.reset_all()` to clear all solvers and stand-alone nets you have created.
