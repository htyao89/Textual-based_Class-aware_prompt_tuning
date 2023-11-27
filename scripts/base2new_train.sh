#!/bin/bash

cd ..

# custom config
DATA=XXXXX
TRAINER=TCP
WEIGHT=1.0

CFG=vit_b16_ep100_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output_1108

for DATASET in caltech101 dtd eurosat fgvc_aircraft food101  oxford_flowers oxford_pets stanford_cars ucf101
do
for SEED in 1 2 3
do
    DIR=${FOLDER}_${NCTX}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done


LOADEP=50
SUB=new
for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${FOLDER}_${NCTX}/base2new/train_base/${COMMON_DIR}
    DIR=${FOLDER}_${NCTX}_eval/base2new/test_${SUB}/${COMMON_DIR}

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
done


for DATASET in sun397 imagenet
do
for SEED in 1 2 3
do
    DIR=${FOLDER}_${NCTX}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W ${WEIGHT} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base
    fi
done


LOADEP=25
SUB=new
for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=${FOLDER}_${NCTX}/base2new/train_base/${COMMON_DIR}
    DIR=${FOLDER}_${NCTX}/base2new/test_${SUB}/${COMMON_DIR}

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
done

