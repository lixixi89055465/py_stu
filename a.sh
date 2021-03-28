#!/usr/bin/expect

set timeout 30 

git add * 
git commit -m 'test' 
spawn git push origin master
expect "Username for 'https://github.com'":
send "89055465ab\r"



