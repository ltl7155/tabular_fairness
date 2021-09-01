for ((i=1;i<21;i++))
do
  for ((j=1;j<20;j++))
  do
#    CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair_adult.py --attr a\&g --p0 $[i] --p1 $[j] --acc_lb 0.825 2>&1 >adult_a\&g.log  &
     CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair/repair_adult.py --attr r  --p0 $[i] --p1 $[j] --ablation 0 --percent 0.3 --weight_threshold 0.5 2>&1 >adult_a\&r.log &
  done
  sleep 2m
  wait
  echo -n "$[i] is finished!";
done