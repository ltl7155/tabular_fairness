#
#if `git st  `|grep -q "modified:"; 
#then 
#    `git st |grep modified |awk '{print $NF}' |xargs git add`
#fi
git st |grep modified


