export CUDA_VISIBLE_DEVICES='1,2,3,4,5'

cd ..

port=12355
datapath=/datassd2/classification/imagenet/train
valpath=/datahdd/imagenet-val/val
ditpath=/datahdd/tfliu/pretrained_models/DiT-XL-2-256x256.pt
proj=df
config=imagewoof
statspth=/datassd2/tfliu/stats/$config


#---------------------making template samples for statistics matching------------------
python make_template_stats_df.py \
    --port $port --config-name $config --dit-path $ditpath \
    --stats-path $statspth --seed 0 --tqdm


#---------------------randomly synthesizing data-free concentrated dataset-----------------------
seed=0
numpiv=10 #IPC
exptag=ipc$numpiv
pivotdir=/datassd2/tfliu/samples/${proj}/${config}/${exptag}/seed${seed}

python synthesize_df.py \
    --port $port --config-name $config --dit-path $ditpath \
    --stats-path $statspth --pivot-dir $pivotdir --seed $seed \
    --num-pivots $numpiv --tqdm --random

#------------------------evaluating the concentrated dataset on resnet_ap10---------------------
cachedir=/datassd2/tfliu/cache/${proj}/${config}/${exptag}/seed${seed}
valseed=0
python train.py \
    --config-name ${config} --pivot-dir ${pivotdir} \
    --val-dir ${valpath} --cache-dir ${cachedir} \
    --image-size 224 --in-size 224 --seed $valseed \
    --net-type resnet_ap --depth 10 --norm-type instance \
    --val-batch-size 100 --batch-size 64 --workers 10 --opt adamw \
    --lr-mul 1 --num-pivots ${numpiv} --print-freq 10 \
    --sche steplr