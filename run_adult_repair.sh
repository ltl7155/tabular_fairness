# for ((i=1;i<21;i++))
# do
#   for ((j=1;j<20;j++))
#   do
# #    CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair_adult.py --attr a\&g --p0 $[i] --p1 $[j] --acc_lb 0.825 2>&1 >adult_a\&g.log  &
#      CUDA_VISIBLE_DEVICES=$[j%7 + 1] nohup python repair/repair_adult.py --attr r  --p0 $[i] --p1 $[j] --ablation 4 --percent 0.3 --weight_threshold 0.2 2>&1 >adult_a\&r.log &
#   done
#   sleep 2m
#   wait
#   echo -n "$[i] is finished!";
# done

for ((j=1;j<4;j++))
do
    CUDA_VISIBLE_DEVICES=$[j%7 + 1] nohup python repair/repair_adult.py --attr r  --ablation j --percent 0.3 --weight_threshold 0.2 2>&1 >adult_a\&r.log &
done
sleep 2m