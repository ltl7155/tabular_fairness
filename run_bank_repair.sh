for ((j=1;j<10;j++))
do
    nohup python repair_bank.py --attr a --p0 $[i] --p1 $[j] --acc_lb 0.88 2>&1 >bank.log &
done
