import os
import shutil


def rename_and_copy_file(src_path, dest_path, filename):
	new_filename = filename.replace("1", "")
	src_file = os.path.join(src_path, filename)
	dest_file = os.path.join(dest_path, new_filename)
	if os.path.exists(dest_file):
		os.remove(dest_file)
	shutil.copy(src_file, dest_file)


def main():
	# 修改并复制 Coordinator.py 文件
	rename_and_copy_file(".\\rope", ".\\rope", "1Models.pyd")

	# 修改并复制 VideoManager.py 文件
	rename_and_copy_file(".\\rope", ".\\rope", "1VideoManager.pyd")

	# 修改并复制 clip.py 文件
	rename_and_copy_file(".\\rope", ".\\rope", "1GUI.pyd")
	# 修改并复制 clip.py 文件
	rename_and_copy_file(".\\rope", ".\\rope", "1Dicts.py")
# 修改并复制 clip.py 文件
	rename_and_copy_file(".\\rope", ".\\rope", "1Coordinator.py")



if __name__ == "__main__":
	main()
	print("cpu启动")
