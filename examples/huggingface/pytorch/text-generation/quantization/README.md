# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for large language models, The following are the models we validated, more models are working in progress.
>**Note**: The default search algorithm is beam search with num_beams = 4, if you'd like to use greedy search for comparison, add "--greedy" in args.


# Prerequisite​
## 1. Create Environment​
Pytorch and Intel-extension-for-pytorch version 2.1 are required, the dependent packages are listed in requirements, we recommend create environment as the following steps.

```bash
pip install -r requirements.txt
```

> Note: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```


# Run
We provide compression technologies such as `MixedPrecision`, `SmoothQuant` and `WeightOnlyQuant` with `RTN/AWQ/TEQ` algorithms and `BitsandBytes`, `load_in_4bit` and `load_in_8bit` work on CPU device, and also support `PEFT` optimized model compression, the followings are command to show how to use it.

## 1. Performance
``` bash
# Please use "--peft_model_id" to replace "--model" if the peft model is used.
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
# fp32
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --benchmark
# mixedprecision
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --mixed_precision \
    --benchmark
# smoothquant
# [alternative] --int8 is used for int8 only, --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --sq \
    --alpha 1.0 \
    --int8 \
    --benchmark
# weightonlyquant
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --woq \
    --benchmark
# load_in_4bit
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_4bit True \
    --benchmark
# load_in_8bit
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_8bit True \
    --benchmark

```

## 2. Accuracy
```bash
# Please use "--peft_model_id" to replace "--model" if the peft model is used.
# fp32
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --accuracy \
    --tasks "lambada_openai"
# mixedprecision
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --mixed_precision \
    --accuracy \
    --tasks "lambada_openai"
# smoothquant
# [alternative] --int8 is used for int8 only, --int8_bf16_mixed is used for int8 mixed bfloat16 precision.
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --sq \
    --alpha 1.0 \
    --int8 \
    --accuracy \
    --tasks "lambada_openai"
# weightonlyquant
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --woq \
    --accuracy \
    --tasks "lambada_openai"
# load_in_4bit
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_4bit True \
    --accuracy \
    --tasks "lambada_openai"
# load_in_8bit
python run_generation.py \
    --model EleutherAI/gpt-j-6b \
    --load_in_8bit True \
    --accuracy \
    --tasks "lambada_openai"

```
