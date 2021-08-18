for ((j=1;j<10;j++))
do
#    CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair_adult.py --attr a\&g --p0 $[i] --p1 $[j] --acc_lb 0.825 2>&1 >adult_a\&g.log  &
    CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair_adult.py --attr g --p1 $[j] --percent 0.3 --weight_threshold 0.2 --acc_lb 0.5 2>&1 >adult_g.log &
done
sleep 2m
wait
