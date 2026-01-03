export CUDA_VISIBLE_DEVICES='1,2,3,4,5'

cd ..

port=12355
datapath=/datassd2/classification/imagenet/train
valpath=/datahdd/imagenet-val/val
ditpath=/datahdd/tfliu/pretrained_models/DiT-XL-2-256x256.pt
proj=da
config=imagewoof
featpth=/datassd2/tfliu/feats/$config

#-----------------------------encode real training data with VAE----------------------
python encode_features_da.py \
    --port $port --config-name $config --data-path ${datapath} \
    --feat-path $featpth --tqdm


numpiv=100 #IPC
exptag=ipc${numpiv}
pivotdir=../samples/${proj}/${config}/${exptag}
cachedir=../cache/${proj}/${config}/${exptag}
strength=0.0005

#---------------------------condense to ${numpiv} IPC-----------------------
python synthesize_da.py \
    --port ${port} \
    --config-name ${config} --dit-path ${ditpath} \
    --feat-path ${featpth} --pivot-dir ${pivotdir} \
    --image-size 256 --num-pivots ${numpiv} \
    --sort-strength ${strength} --tqdm

#--------------------------evaluating the synthetic dataset---------------
valseed=0
python train.py \
    --config-name ${config} --pivot-dir ${pivotdir} \
    --val-dir ${valpath} --cache-dir ${cachedir} \
    --image-size 224 --in-size 224 --seed $valseed \
    --net-type resnet_ap --depth 10 --norm-type instance \
    --val-batch-size 100 --batch-size 64 --workers 10 --opt adamw \
    --lr-mul 1 --num-pivots ${numpiv} --print-freq 10 \
    --sche steplr

#--------------------------train doping model with the synthetic dataset---------------
valseed=0
python train.py \
    --config-name ${config} --pivot-dir ${pivotdir} \
    --val-dir ${valpath} --cache-dir ${cachedir} \
    --image-size 224 --in-size 224 --seed $valseed \
    --net-type resnet_ap --depth 10 --norm-type instance \
    --val-batch-size 100 --batch-size 64 --workers 10 --opt adamw \
    --lr-mul 1 --num-pivots ${numpiv} --print-freq 10 \
    --sche cos --save-ckpt --teach

#-----------------------------doping with 500IPC-------------------------------
numnew=500
newdir=/datassd2/samples/new/${proj}/${exptag}_new${numnew}

python mix_from_real.py --port ${port} \
    --config-name ${config} --data-path ${datapath} \
    --pivot-path ${pivotdir} --new-dir ${newdir} \
    --cache-dir ${cachedir} --net-type resnet_ap --depth 10 --norm-type instance \
    --num-new ${numnew} --save-odds-to ${cachedir} --copy-pivots

#--------------------------evaluating the concentrated dataset---------------
python train.py \
    --config-name ${config} --pivot-dir ${newdir} \
    --val-dir ${valpath} --cache-dir ${cachedir} \
    --image-size 224 --in-size 224 --seed $valseed \
    --net-type resnet_ap --depth 10 --norm-type instance \
    --val-batch-size 100 --batch-size 64 --workers 10 --opt adamw \
    --lr-mul 1 --num-pivots ${numpiv} --print-freq 10 \
    --sche cos --save-ckpt --teach

#-----------------------------doping with 600IPC after previous doping-------------------------------
numnew=600
newdir=/datassd2/samples/new/${proj}/${exptag}_new${numnew}

python mix_from_real.py --port ${port} \
    --config-name ${config} --data-path ${datapath} \
    --pivot-path ${pivotdir} --new-dir ${newdir} \
    --cache-dir ${cachedir} --net-type resnet_ap --depth 10 --norm-type instance \
    --num-new ${numnew} --load-odds-from ${cachedir} --copy-pivots \
    --image-size 256 --in-size 224

#--------------------------train a model with the concentrated dataset---------------
python train.py \
    --config-name ${config} --pivot-dir ${newdir} \
    --val-dir ${valpath} --cache-dir ${cachedir} \
    --image-size 224 --in-size 224 --seed $valseed \
    --net-type resnet_ap --depth 10 --norm-type instance \
    --val-batch-size 100 --batch-size 64 --workers 10 --opt adamw \
    --lr-mul 1 --num-pivots ${numpiv} --print-freq 10 \
    --sche cos --save-ckpt --teach
