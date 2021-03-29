#!/usr/bin/expect

set timeout 30 

spawn git add * 
spawn git commit -m 'test' 
spawn git push origin master
expect "*Username*"
send "1850094299@qq.com\r"
expect "*Password*"
send "89055465ab\r" 
exp_continue



