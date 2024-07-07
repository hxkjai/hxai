import os
import re
import socket
import subprocess
import sys

import psutil


def check_ffmpeg_path():
    cwd = os.getcwd()
    ffmpeg_path = os.path.join(cwd, "ffmpeg", "bin", "ffmpeg.exe")
    ffprobe_path = os.path.join(cwd, "ffmpeg", "bin", "ffprobe.exe")
    return os.path.isfile(ffmpeg_path) and os.path.isfile(ffprobe_path)

def get_default_browser():
    browser = ""
    try:
        cmd = r'reg query HKEY_CURRENT_USER\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice /v ProgId'
        output = subprocess.check_output(cmd, shell=True).decode()
        browser = output.split()[-1].split('\\')[-1]
    except Exception as e:
        print(f"Error: {e}")
        browser = "Unknown"
    return browser

def get_hostname():
    try:
        hostname = socket.gethostname()
        return hostname
    except socket.error as e:
        print("Error: ", e)

def get_pagefile_size():
    try:
        pagefile = psutil.swap_memory().total
    except UnboundLocalError:
        pagefile = 0
    except RuntimeError:
        pagefile = None
    return pagefile
    
    
def main():
    allowed_pattern = re.compile(r'^[a-zA-Z0-9_@#$%^&()_+\-=\s\.]*$')
    hostname = get_hostname()
    pagefile = get_pagefile_size()
    default_browser = get_default_browser() 
    if check_ffmpeg_path():
        print("FFmpeg already installed, skipping...")
    else:
        try:
            sys.exit(0)
        finally:
            print("未找到FFmpeg，整合包可能不完整，请重新下载")
    if pagefile is None:
        print("WARNING | 系统未开启性能计数器，无法获取当前虚拟内存状态，确保虚拟内存大于30G后可忽略此警告")
    elif pagefile < 31457280000: # 30 GiB
        print("WARNING | 虚拟内存不足30GB，可能会导致使用问题，请将虚拟内存设置为至少30G")
    if "chrome" not in default_browser.lower() and "edge" not in default_browser.lower() and "firefox" not in default_browser.lower():
        print("WARNING | 默认浏览器不符合要求，可能会影响使用，请更换为Chrome或Edge浏览器")
    if not allowed_pattern.match(hostname):
        print("WARNING | 计算机主机名中含有非西文字符，启动Tensorboard时可能出错，请在计算机设置中修改")

    os.system("workenv\python.exe app.py")
            
if __name__ == "__main__":
    main()