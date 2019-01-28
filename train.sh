# original kdeep 
python train.py --dataroot ~/work/pdbbind/2018/refined-set \
                --csvfile ../data/refined_set.csv --model kdeep \
                --gpu_ids 0 --batch_size 256 --nThreads 16 \
                --lr 0.0001 --niter 5 --niter_decay 5 \
                --channels cno --rvdw 2 \
                --save_epoch_freq 5 --init_type kaiming
