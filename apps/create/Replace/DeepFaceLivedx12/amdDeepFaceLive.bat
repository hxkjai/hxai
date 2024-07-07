@echo off
cd /D %~dp0
call _internal\setenv.bat
"%PYTHONEXECUTABLE%" _internal\DeepFaceLive\main.py run DeepFaceLive --userdata-dir="%~dp0userdata" --no-cuda
pause
