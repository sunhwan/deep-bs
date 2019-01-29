# DEEP BINDING SITE

To train:

    python train.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                    --csvfile /home/sunhwan/work/pdbbind/deep/data/refined_set.csv \
                    --model keep --gpu_ids 0 --batch_size 256 --nThreads 16 \
                    --lr 0.0001 --niter 50 --niter_decay 25 \
                    --channels cno --rvdw 2 --save_epoch_freq 5 --continue_train 

To test:

    python test.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                   --csvfile /home/sunhwan/work/pdbbind/deep/data/test_set.csv \
                   --model kdeep --gpu_ids 3 --batch_size 256 --nThreads 16 

# Preprocess PdbBind dataset

To make reading data faster, use the following command to preprocess PDB/Mol2 files and determine
Smina atom types prior to training.

    python data.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set
    python data.py --dataroot /home/sunhwan/work/pdbbind/2018/other-set
