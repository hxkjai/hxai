@echo off
chcp 65001
echo --环线科技--技术问题微信18064884865--

:menu
echo 请选择一个模块:

echo 0. 打开官网链接

echo 1. OpenXLab在线体验

echo 2. Hugging Face 在线体验

echo 3. 启动本地ProPainter  提示：把启动名字改成ProPainter.bat  就能自动检测到

set /p option=Enter option (请输入一个数字)：
echo "%option%"

if "%option%"=="0" (
   start https://github.com/sczhou/ProPainter
) else if "%option%"=="1" (
   start https://openxlab.org.cn/apps/detail/ShangchenZhou/ProPainter
) else if "%option%"=="2" (
   start https://huggingface.co/spaces/sczhou/ProPainter
) else if "%option%"=="3" (
   call .\ProPainter.bat
) else (
   echo **无效的选择，请再次输入**
   goto menu
)