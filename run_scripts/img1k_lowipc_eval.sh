export CUDA_VISIBLE_DEVICES='2'

cd ../../eval_rded

arch=resnet18

echo --------------------------${arch}-50IPC-------------------
python validate.py \
    --syn-data-path /datassd/bases/imagenet1k/ipc50 --ipc 50 \
    --val-dir /datahdd/imagenet-val/val --subset imagenet-1k --workers 8 \
    --stud-name ${arch}
