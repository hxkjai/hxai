@echo off
chcp 65001
setlocal

set "InstallDir=%cd%\topazphoto"
set "ModelDir=%InstallDir%"

echo Windows Registry Editor Version 5.00 > path.reg
echo. >> path.reg
echo [HKEY_LOCAL_MACHINE\SOFTWARE\Topaz Labs LLC\Topaz Photo AI] >> path.reg
echo "InstallDir"="%InstallDir:\=\\%\\" >> path.reg
echo "ModelDir"="%ModelDir:\=\\%\\" >> path.reg

echo.
echo 文件 path.reg 已生成。双击运行添加到注册表
echo.
pause

