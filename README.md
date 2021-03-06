# DEEP BINDING SITE

## KDEEP Model

To train:

    python train.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                    --csvfile /home/sunhwan/work/pdbbind/deep/data/train.csv \
                    --gpu_ids 0 --batch_size 64 --nThreads 16 --init_type kaiming \
                    --lr 0.0001 --niter 50 --niter_decay 25 --save_epoch_freq 5 \
                    --model kdeep --grid_method kdeep --grid_size 24 --grid_spacing 1.0 \
                    --channels kdeep --rvdw 2

add `--continue-train` to resume training from the latest weight.

To test:

    python test.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                   --csvfile /home/sunhwan/work/pdbbind/deep/data/test.csv \
                   --gpu_ids 0 --batch_size 64 --nThreads 16 \
                   --model kdeep --grid_method kdeep --grid_size 24 --grid_spacing 1.0 \
                   --channels kdeep --rvdw 2

## GNINA Model

To train:

    python train.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                    --csvfile /home/sunhwan/work/pdbbind/deep/data/train.csv \
                    --gpu_ids 0 --batch_size 32 --nThreads 16 --init_type kaiming \
                    --lr 0.0001 --niter 50 --niter_decay 25 --save_epoch_freq 5 \
                    --model gnina --grid_method gnina --grid_size 48 --grid_spacing 0.5 \
                    --channels gnina

add `--continue-train` to resume training from the latest weight.

To test:

    python test.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                   --csvfile /home/sunhwan/work/pdbbind/deep/data/test.csv \
                   --gpu_ids 0 --batch_size 32 --nThreads 16 \
                   --model gnina --grid_method gnina --grid_size 48 --grid_spacing 0.5 \
                   --channels gnina

## GNINA with docked pose Model

To train:

    python train.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                    --csvfile /home/sunhwan/work/pdbbind/deep/data/train.csv \
                    --gpu_ids 0 --batch_size 32 --nThreads 16 --init_type kaiming \
                    --lr 0.0001 --niter 50 --niter_decay 25 --save_epoch_freq 5 \
                    --model gnina_docked --grid_method gnina_docked --grid_size 48 \
                    --grid_spacing 0.5 --channels gnina --dataset_mode pdbbind_docked

add `--continue-train` to resume training from the latest weight.

To test:

    python test.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                   --csvfile /home/sunhwan/work/pdbbind/deep/data/test.csv \
                   --gpu_ids 0 --batch_size 32 --nThreads 16 \
                   --model gnina --grid_method gnina --grid_size 48 --grid_spacing 0.5 \
                   --channels gnina


# Preprocess PdbBind dataset

To make reading data faster, use the following command to preprocess PDB/Mol2 files and determine
Smina atom types prior to training.

    python data.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set
    python data.py --dataroot /home/sunhwan/work/pdbbind/2018/other-set
