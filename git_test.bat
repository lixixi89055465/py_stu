@echo off
git add .
git commit -m "%1"//待传入的参数
git push
echo push respostory successfully
pause