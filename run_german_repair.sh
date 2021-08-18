for ((i=1;i<11;i++))
do
  for ((j=1;j<10;j++))
  do
      CUDA_VISIBLE_DEVICES=$[j-1] nohup python repair_german.py --attr a --p0 $[i] --p1 $[j] --acc_lb 0.7 --percent 0.3 --weight_threshold 0.2 2>&1 >german.log &
#      CUDA_VISIBLE_DEVICES=$[j-1] nohup  python repair_german.py --attr g --p0 $[i] --p1 $[j] --acc_lb 0.7 --percent 0.3 --weight_threshold 0.2 2>&1 >german.log &
#      CUDA_VISIBLE_DEVICES=$[j%8] nohup  python repair_german.py --attr g\&a --p0 $[i] --p1 $[j] --acc_lb 0.7 --percent 0.3 --weight_threshold 0.2 2>&1 >german.log &
  done
  sleep 8m
  wait
  echo -n "$[i] is finished!";
done