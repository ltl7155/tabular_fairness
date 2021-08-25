cd /mnt/mfs/litl/tabular_fairness/FairnessRepair && \
sh auto_commit_add.sh 2>&1 >/dev/null  && \
sh auto_commit_del.sh 2>&1 >/dev/null  && \
git ci -m 'auto commit' && \
git push origin main && \
echo "finish " 


