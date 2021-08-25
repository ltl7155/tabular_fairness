git st |grep deleted  |awk '{print $NF}' |xargs git rm 


