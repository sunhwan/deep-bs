# DEEP BINDING SITE

To train:

    python deep/train.py --dataroot /home/sunhwan/work/pdbbind/2018/refined-set \
                         --csvfile /home/sunhwan/work/pdbbind/deep/data/refined_set.csv \
                         --model keep \
                         --gpu_ids 0 --batch_size 256 --nThreads 0 --niter 1 --niter_decay 1
