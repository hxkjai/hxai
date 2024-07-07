@echo off
chcp 65001

runtime\python.exe gui_v1.py

if %errorlevel% EQU 0 (
  echo.
  echo 程序运行正常！
) else (
  echo.
  echo 正在检查 FreeSimpleGUI 模块...
  runtime\python.exe -c "import FreeSimpleGUI; print('FreeSimpleGUI 模块存在')" > nul 2>&1
  if errorlevel 1 (
    echo.
    echo 正在更新自动安装完成后重开应用 FreeSimpleGUI...
    echo.
    runtime\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple FreeSimpleGUI
    echo.
    echo 安装完成，请重新运行程序！
  ) else (
    echo.
    echo FreeSimpleGUI 模块已存在，但程序运行失败。
  )
)