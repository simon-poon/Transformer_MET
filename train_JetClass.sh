#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=50
dataopts="--num-workers 2 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 256 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi


# currently only Pythia
SAMPLE_TYPE=Pythia

$CMD \
    --data-train \
    "${DATADIR}/perfNano_TTbar_PU200.110X_set0.root" \
    "${DATADIR}/perfNano_TTbar_PU200.110X_set1.root" \
    "${DATADIR}/perfNano_TTbar_PU200.110X_set2.root" \
    "${DATADIR}/perfNano_TTbar_PU200.110X_set3.root" \
    "${DATADIR}/perfNano_TTbar_PU200.110X_set4.root" \
    --data-val "${DATADIR}/perfNano_TTbar_PU200.110X_set5.root" \
    --data-test \
    "${DATADIR}/perfNano_TTbar_PU200.110X_set6.root" \
    --data-config data/JetClass/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix /mettransformervol/saved_models/mettransformer_test/ \
    $dataopts $batchopts \
    --num-epochs $epochs --gpus 0 \
    --optimizer adam --log /mettransformervol/logs/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}${suffix} \
    --regression-mode \
    "${@:3}"
