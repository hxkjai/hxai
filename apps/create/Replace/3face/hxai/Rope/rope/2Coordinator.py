# #!/usr/bin/env python3

import time
import torch
from torchvision import transforms

import rope.GUI as GUI
import rope.VideoManager as VM
import rope.Models as Models
from rope.external.clipseg import CLIPDensePredT

resize_delay = 1
mem_delay = 1

# @profile
def coordinator():
    global gui, vm, action, frame, r_frame, load_notice, resize_delay, mem_delay
    # start = time.time()


    if gui.get_action_length() > 0:
        action.append(gui.get_action())
    if vm.get_action_length() > 0:
        action.append(vm.get_action())
##################
    if vm.get_frame_length() > 0:
        frame.append(vm.get_frame())

    if len(frame) > 0:
        gui.set_image(frame[0], False)
        frame.pop(0)
 ####################
    if vm.get_requested_frame_length() > 0:
        r_frame.append(vm.get_requested_frame())
    if len(r_frame) > 0:
        gui.set_image(r_frame[0], True)
        r_frame=[]
 ####################
    if len(action) > 0:

        if action[0][0] == "load_target_video":
            vm.stop_camera_video()
            vm.load_target_video(action[0][1])
            action.pop(0)
        elif action[0][0] == "rotate_image":
            print("Received 'rotate_image' action.")  # 打印信息确认接收到动作
            print(f"Action: {action[0][0]}, Parameter: {action[0][1]}")
            print(f"Current RotateButton state: {vm.control['RotateButton']}")  # 打印 RotateButton 状态
            vm.rotation_angle = (vm.rotation_angle + 90) % 360
            print("Rotating image. Angle:", vm.rotation_angle, "degrees")
            # 处理 rotate_image 动作
            if gui.widget['PreviewModeTextSel'].get() == 'Video':
                vm.get_requested_video_frame(gui.video_slider.get())  # 请求当前帧
            elif gui.widget['PreviewModeTextSel'].get() == 'Image':
                vm.get_requested_image()  # 请求当前图像
            action.pop(0)

        elif action[0][0] == 'control':
            vm.control = action[0][1]  # 更新 VideoManager 的 control 字典
            action.pop(0)






        elif action[0][0] == "load_target_image":
            vm.stop_camera_video()
            vm.load_target_image(action[0][1])
            action.pop(0)
        elif action[0][0] == "play_video":
            vm.play_video(action[0][1])
            action.pop(0)
        elif action[0][0] == "get_requested_video_frame":
            vm.get_requested_video_frame(action[0][1], marker=True)
            action.pop(0)
        elif action[0][0] == "get_requested_video_frame_without_markers":
            vm.get_requested_video_frame(action[0][1], marker=False)
            action.pop(0)
        elif action[0][0] == "get_requested_image":
            vm.get_requested_image()
            action.pop(0)
        # elif action[0][0] == "swap":
        #     vm.swap = action[0][1]
        #     action.pop(0)
        elif action[0][0] == "target_faces":
            vm.assign_found_faces(action[0][1])
            action.pop(0)
        elif action [0][0] == "saved_video_path":
            vm.saved_video_path = action[0][1]
            action.pop(0)
        elif action [0][0] == "vid_qual":
            vm.vid_qual = int(action[0][1])
            action.pop(0)
        elif action [0][0] == "set_stop":
            vm.stop_marker = action[0][1]
            action.pop(0)
        elif action [0][0] == "perf_test":
            vm.perf_test = action[0][1]
            action.pop(0)
        elif action [0][0] == 'ui_vars':
            vm.ui_data = action[0][1]
            action.pop(0)
        elif action [0][0] == 'control':
            vm.control = action[0][1]
            action.pop(0)
        elif action [0][0] == "parameters":
            if action[0][1]["CLIPSwitch"]:
                if not vm.clip_session:
                    vm.clip_session = load_clip_model()

            vm.parameters = action[0][1]
            action.pop(0)
        elif action [0][0] == "markers":
            vm.markers = action[0][1]
            action.pop(0)


        elif action[0][0] == "function":
            eval(action[0][1])
            action.pop(0)
        elif action [0][0] == "clear_mem":
            vm.clear_mem()
            action.pop(0)


        # From VM
        elif action[0][0] == "stop_play":
            gui.set_player_buttons_to_inactive()
            action.pop(0)

        elif action[0][0] == "set_slider_length":
            gui.set_video_slider_length(action[0][1])
            action.pop(0)




        elif action[0][0] == "load_live_stream":  # 添加处理 load_live_stream 动作
            vm.load_live_stream()
            action.pop(0)
        elif action[0][0] == "load_camera":
            vm.load_camera(action[0][1])
            action.pop(0)
        elif action[0][0] == "start_virtual_camera":
            print("start")
            vm.start_virtual_camera()
            action.pop(0)
        elif action[0][0] == "stop_virtual_camera":
            vm.stop_virtual_camera()
            action.pop(0)



        else:
            print("Action not found: "+action[0][0]+" "+str(action[0][1]))
            action.pop(0)




    if resize_delay > 100:
        gui.check_for_video_resize()
        resize_delay = 0
    else:
        resize_delay +=1

    if mem_delay > 1000:
        gui.update_vram_indicator()
        mem_delay = 0
    else:
        mem_delay +=1

    vm.process()
    gui.after(1, coordinator)
    # print(time.time() - start)





def load_clip_model():
    # https://github.com/timojl/clipseg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    # clip_session = CLIPDensePredTMasked(version='ViT-B/16', reduce_dim=64)
    clip_session.eval();
    clip_session.load_state_dict(torch.load('../models/models1/rd64-uni-refined.pth'), strict=False)
    clip_session.to(device)
    return clip_session




def run():
    global gui, vm, action, frame, r_frame, resize_delay, mem_delay

    models = Models.Models()  # 创建模型对象
    gui = GUI.GUI(models)
    vm = VM.VideoManager(models)

    def on_closing():
        global coordinator  # 将 coordinator 声明为全局变量
        gui.destroy()
        vm.stop_camera_video()
        # 停止 coordinator 函数的执行
        gui.after_cancel(coordinator)

    gui.protocol("WM_DELETE_WINDOW", on_closing)

    action = []
    frame = []
    r_frame = []

    gui.initialize_gui()
    coordinator()  # 启动 coordinator 函数

    gui.mainloop()  # 运行 GUI 事件循环

    # 释放模型资源
    models.delete_models()  # 在 GUI 事件循环结束后释放模型对象

if __name__ == "__main__":
    run()