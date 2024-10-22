import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import re
import time

import pandas
import numpy as np
from tqdm import tqdm
import random
import os
import gradio as gr
import json
from utils import combine_audio, save_audio, batch_split, normalize_zh
from tts_model import load_chat_tts_model, clear_cuda_cache, deterministic, generate_audio_for_seed

parser = argparse.ArgumentParser(description="Gradio ChatTTS MIX")
parser.add_argument("--source", type=str, default="huggingface", help="Model source: 'huggingface' or 'local'.")
parser.add_argument("--local_path", type=str, help="Path to local model if source is 'local'.")
parser.add_argument("--share", default=False, action="store_true", help="Share the server publicly.")

args = parser.parse_args()

# 存放音频种子文件的目录
SAVED_DIR = "saved_seeds"

# mkdir
if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)

# 文件路径
SAVED_SEEDS_FILE = os.path.join(SAVED_DIR, "saved_seeds.json")

# 选中的种子index
SELECTED_SEED_INDEX = -1

# 初始化JSON文件
if not os.path.exists(SAVED_SEEDS_FILE):
    with open(SAVED_SEEDS_FILE, "w") as f:
        f.write("[]")

chat = load_chat_tts_model(source=args.source, local_path=args.local_path)
# chat = None
# chat = load_chat_tts_model(source="local", local_path="models")

# 抽卡的最大数量
max_audio_components = 10


# print("loading ChatTTS model...")
# chat = ChatTTS.Chat()
# chat.load_models(source="local", local_path="models")
# torch.cuda.empty_cache()


# 加载
def load_seeds():
    with open(SAVED_SEEDS_FILE, "r") as f:
        global saved_seeds
        saved_seeds = json.load(f)
    return saved_seeds


def display_seeds():
    seeds = load_seeds()
    # 转换为 List[List] 的形式
    return [[i, s['seed'], s['name']] for i, s in enumerate(seeds)]


saved_seeds = load_seeds()
num_seeds_default = 2


def save_seeds():
    global saved_seeds
    with open(SAVED_SEEDS_FILE, "w") as f:
        json.dump(saved_seeds, f)
    saved_seeds = load_seeds()


# 添加 seed
def add_seed(seed, name, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            return False
    saved_seeds.append({
        'seed': seed,
        'name': name
    })
    if save:
        save_seeds()


# 修改 seed
def modify_seed(seed, name, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            s['name'] = name
            if save:
                save_seeds()
            return True
    return False


def delete_seed(seed, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            saved_seeds.remove(s)
            if save:
                save_seeds()
            return True
    return False


def generate_seeds(num_seeds, texts, tq):
    """
    生成随机音频种子并保存
    :param num_seeds:
    :param texts:
    :param tq:
    :return:
    """
    seeds = []
    sample_rate = 24000
    # 按行分割文本 并正则化数字和标点字符
    texts = [normalize_zh(_) for _ in texts.split('\n') if _.strip()]
    print(texts)
    if not tq:
        tq = tqdm
    for _ in tq(range(num_seeds), desc=f"随机音色生成中..."):
        seed = np.random.randint(0, 9999)

        filename = generate_audio_for_seed(chat, seed, texts, 1, 5, "[oral_2][laugh_0][break_4]", 0.3, 0.7, 20)
        seeds.append((filename, seed))
        clear_cuda_cache()

    return seeds


# 保存选定的音频种子
def do_save_seed(seed):
    seed = seed.replace('保存种子 ', '').strip()
    if not seed:
        return
    add_seed(int(seed), seed)
    gr.Info(f"Seed {seed} has been saved.")


def do_save_seeds(seeds):
    assert isinstance(seeds, pandas.DataFrame)

    seeds = seeds.drop(columns=['Index'])

    # 将 DataFrame 转换为字典列表格式，并将键转换为小写
    result = [{k.lower(): v for k, v in row.items()} for row in seeds.to_dict(orient='records')]
    print(result)
    if result:
        global saved_seeds
        saved_seeds = result
        save_seeds()
        gr.Info(f"Seeds have been saved.")
    return result


def do_delete_seed(val):
    # 从 val 匹配 [(\d+)] 获取index
    index = re.search(r'\[(\d+)\]', val)
    global saved_seeds
    if index:
        index = int(index.group(1))
        seed = saved_seeds[index]['seed']
        delete_seed(seed)
        gr.Info(f"Seed {seed} has been deleted.")
    return display_seeds()


def seed_change_btn():
    global SELECTED_SEED_INDEX
    if SELECTED_SEED_INDEX == -1:
        return '删除'
    return f'删除 idx=[{SELECTED_SEED_INDEX[0]}]'


def audio_interface(num_seeds, texts, progress=gr.Progress()):
    """
    生成音频
    :param num_seeds:
    :param texts:
    :param progress:
    :return:
    """
    seeds = generate_seeds(num_seeds, texts, progress.tqdm)
    wavs = [_[0] for _ in seeds]
    seeds = [f"保存种子 {_[1]}" for _ in seeds]
    # 不足的部分
    all_wavs = wavs + [None] * (max_audio_components - len(wavs))
    all_seeds = seeds + [''] * (max_audio_components - len(seeds))
    return [item for pair in zip(all_wavs, all_seeds) for item in pair]


def audio_interface_empty(num_seeds, texts, progress=gr.Progress(track_tqdm=True)):
    return [None, ""] * max_audio_components


def update_audio_components(slider_value):
    # 根据滑块的值更新 Audio 组件的可见性
    k = int(slider_value)
    audios = [gr.Audio(visible=True)] * k + [gr.Audio(visible=False)] * (max_audio_components - k)
    tbs = [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (max_audio_components - k)
    print(f'k={k}, audios={len(audios)}')
    return [item for pair in zip(audios, tbs) for item in pair]


def seed_change(evt: gr.SelectData):
    # print(f"You selected {evt.value} at {evt.index} from {evt.target}")
    global SELECTED_SEED_INDEX
    SELECTED_SEED_INDEX = evt.index
    return evt.index


def generate_tts_audio(text_file, num_seeds, seed, speed, oral, laugh, bk, min_length, batch_size, temperature, top_P,
                       top_K, progress=gr.Progress()):
    from tts_model import generate_audio_for_seed
    from utils import split_text
    if seed in [0, -1, None]:
        seed = random.randint(1, 9999)
    content = ''
    if os.path.isfile(text_file):
        content = ""
    elif isinstance(text_file, str):
        content = text_file
    texts = split_text(content, min_length=min_length)
    print(texts)

    if oral < 0 or oral > 9 or laugh < 0 or laugh > 2 or bk < 0 or bk > 7:
        raise ValueError("oral_(0-9), laugh_(0-2), break_(0-7) out of range")

    refine_text_prompt = f"[oral_{oral}][laugh_{laugh}][break_{bk}]"
    try:
        output_files = generate_audio_for_seed(chat, seed, texts, batch_size, speed, refine_text_prompt, temperature,
                                               top_P, top_K, progress.tqdm)
        return output_files
    except Exception as e:
        return str(e)


def generate_seed():
    new_seed = random.randint(1, 9999)
    return {
        "__type__": "update",
        "value": new_seed
    }


def update_label(text):
    word_count = len(text)
    return gr.update(label=f"朗读文本（字数: {word_count}）")


with gr.Blocks() as demo:
    with gr.Tab("音色抽卡"):
        with gr.Row():
            with gr.Column(scale=1):
                texts = [
                    "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。",
                ]
                # gr.Markdown("### 随机音色抽卡")
                gr.Markdown("""
                在相同的 seed 和 温度等参数下，音色具有一定的一致性。点击下面的“随机音色生成”按钮将生成多个 seed。找到满意的音色后，点击音频下方“保存”按钮。
                **注意：不同机器使用相同种子生成的音频音色可能不同，同一机器使用相同种子多次生成的音频音色也可能变化。**
                """)
                input_text = gr.Textbox(label="测试文本",
                                        info="**每行文本**都会生成一段音频，最终输出的音频是将这些音频段合成后的结果。建议使用**多行文本**进行测试，以确保音色稳定性。",
                                        lines=4, placeholder="请输入文本...", value='\n'.join(texts))

                num_seeds = gr.Slider(minimum=1, maximum=max_audio_components, step=1, label="seed生成数量",
                                      value=num_seeds_default)

                generate_button = gr.Button("随机音色抽卡🎲", variant="primary")

                # 保存的种子
                gr.Markdown("### 种子管理界面")
                seed_list = gr.DataFrame(
                    label="种子列表",
                    headers=["Index", "Seed", "Name"],
                    datatype=["number", "number", "str"],
                    interactive=True,
                    col_count=(3, "fixed"),
                    value=display_seeds()
                )
                with gr.Row():
                    refresh_button = gr.Button("刷新")
                    save_button = gr.Button("保存")
                    del_button = gr.Button("删除")
                # 绑定按钮和函数
                refresh_button.click(display_seeds, outputs=seed_list)
                seed_list.select(seed_change).success(seed_change_btn, outputs=[del_button])
                save_button.click(do_save_seeds, inputs=[seed_list], outputs=None)
                del_button.click(do_delete_seed, inputs=del_button, outputs=seed_list)

            with gr.Column(scale=1):
                audio_components = []
                for i in range(max_audio_components):
                    visible = i < num_seeds_default
                    a = gr.Audio(f"Audio {i}", visible=visible)
                    t = gr.Button(f"Seed", visible=visible)
                    t.click(do_save_seed, inputs=[t], outputs=None).success(display_seeds, outputs=seed_list)
                    audio_components.append(a)
                    audio_components.append(t)

                num_seeds.change(update_audio_components, inputs=num_seeds, outputs=audio_components)

                # output = gr.Column()
                # audio = gr.Audio(label="Output Audio")

            generate_button.click(
                audio_interface_empty,
                inputs=[num_seeds, input_text],
                outputs=audio_components
            ).success(audio_interface, inputs=[num_seeds, input_text], outputs=audio_components)
    with gr.Tab("长音频生成"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 文本")
                # gr.Markdown("请上传要转换的文本文件（.txt 格式）。")
                # text_file_input = gr.File(label="文本文件", file_types=[".txt"])
                default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"
                text_file_input = gr.Textbox(label=f"朗读文本（字数: {len(default_text)}）", lines=4,
                                             placeholder="Please Input Text...", value=default_text)
                # 当文本框内容发生变化时调用 update_label 函数
                text_file_input.change(update_label, inputs=text_file_input, outputs=text_file_input)

            with gr.Column():
                gr.Markdown("### 配置参数")
                gr.Markdown("根据需要配置以下参数来生成音频。")
                with gr.Row():
                    num_seeds_input = gr.Number(label="生成音频的数量", value=1, precision=0, visible=False)
                    seed_input = gr.Number(label="指定种子（留空则随机）", value=None, precision=0)
                    generate_audio_seed = gr.Button("\U0001F3B2")

                with gr.Row():
                    speed_input = gr.Slider(label="语速", minimum=1, maximum=10, value=5, step=1)
                    oral_input = gr.Slider(label="口语化", minimum=0, maximum=9, value=2, step=1)

                    laugh_input = gr.Slider(label="笑声", minimum=0, maximum=2, value=0, step=1)
                    bk_input = gr.Slider(label="停顿", minimum=0, maximum=7, value=4, step=1)
                # gr.Markdown("### 文本参数")
                with gr.Row():
                    min_length_input = gr.Number(label="文本分段长度", info="大于这个数值进行分段", value=120,
                                                 precision=0)
                    batch_size_input = gr.Number(label="批大小", info="同时处理的批次 越高越快 太高爆显存", value=5,
                                                 precision=0)
                with gr.Accordion("其他参数", open=False):
                    with gr.Row():
                        # 温度 top_P top_K
                        temperature_input = gr.Slider(label="温度", minimum=0.01, maximum=1.0, step=0.01, value=0.3)
                        top_P_input = gr.Slider(label="top_P", minimum=0.1, maximum=0.9, step=0.05, value=0.7)
                        top_K_input = gr.Slider(label="top_K", minimum=1, maximum=20, step=1, value=20)
                        # reset 按钮
                        reset_button = gr.Button("重置")

        with gr.Row():
            generate_button = gr.Button("生成音频", variant="primary")

        with gr.Row():
            output_audio = gr.Audio(label="生成的音频文件")

        generate_audio_seed.click(generate_seed,
                                  inputs=[],
                                  outputs=seed_input)

        # 重置按钮 重置温度等参数
        reset_button.click(
            lambda: [0.3, 0.7, 20],
            inputs=None,
            outputs=[temperature_input, top_P_input, top_K_input]
        )

        generate_button.click(
            fn=generate_tts_audio,
            inputs=[
                text_file_input,
                num_seeds_input,
                seed_input,
                speed_input,
                oral_input,
                laugh_input,
                bk_input,
                min_length_input,
                batch_size_input,
                temperature_input,
                top_P_input,
                top_K_input,
            ],
            outputs=[output_audio]
        )
    with gr.Tab("角色扮演"):
        def txt_2_script(text):
            lines = text.split("\n")
            data = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("::")
                if len(parts) != 2:
                    continue
                data.append({
                    "character": parts[0],
                    "txt": parts[1]
                })
            return data


        def script_2_txt(data):
            assert isinstance(data, list)
            result = []
            for item in data:
                txt = item['txt'].replace('\n', ' ')
                result.append(f"{item['character']}::{txt}")
            return "\n".join(result)


        def get_characters(lines):
            assert isinstance(lines, list)
            characters = list([_["character"] for _ in lines])
            unique_characters = list(dict.fromkeys(characters))
            print([[character, 0] for character in unique_characters])
            return [[character, 0] for character in unique_characters]


        def get_txt_characters(text):
            return get_characters(txt_2_script(text))


        def llm_change(model):
            llm_setting = {
                "gpt-3.5-turbo-0125": ["https://api.openai.com/v1"],
                "gpt-4o": ["https://api.openai.com/v1"],
                "deepseek-chat": ["https://api.deepseek.com"],
                "yi-large": ["https://api.lingyiwanwu.com/v1"],
                "阿里": ["https://dashscope.aliyuncs.com/compatible-mode/v1"]
            }
            if model in llm_setting:
                return llm_setting[model][0]
            else:
                gr.Error("Model not found.")
                return None


        def ai_script_generate(model, api_base, api_key, text, progress=gr.Progress(track_tqdm=True)):
            from llm_utils import llm_operation
            from config import LLM_PROMPT
            scripts = llm_operation(api_base, api_key, model, LLM_PROMPT, text, required_keys=["txt", "character"])
            return script_2_txt(scripts)


        def generate_script_audio(text, models_seeds, progress=gr.Progress()):
            scripts = txt_2_script(text)  # 将文本转换为剧本
            characters = get_characters(scripts)  # 从剧本中提取角色

            #
            import pandas as pd
            from collections import defaultdict
            import itertools
            from tts_model import generate_audio_for_seed
            from utils import combine_audio, save_audio, normalize_zh
            from config import DEFAULT_BATCH_SIZE, DEFAULT_SPEED, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P

            assert isinstance(models_seeds, pd.DataFrame)

            # 批次处理函数
            def batch(iterable, batch_size):
                it = iter(iterable)
                while True:
                    batch = list(itertools.islice(it, batch_size))
                    if not batch:
                        break
                    yield batch

            models_seeds = models_seeds.to_dict(orient='records')

            # 检查每个角色是否都有对应的种子
            for character, _ in characters:
                if not any(seed['Character'] == character for seed in models_seeds):
                    gr.Info(f"角色 {character} 没有种子，请先设置种子。")
                    return None

            # 将角色和对应的种子存为字典
            character_seeds = {character: [seed['Seed'] for seed in models_seeds if seed['Character'] == character][0]
                               for character, _ in characters}
            # todo 可以自定义 最好是按角色
            refine_text_prompt = "[oral_2][laugh_0][break_4]"
            all_wavs = []

            # 按角色分组，加速推理
            grouped_lines = defaultdict(list)
            for line in scripts:
                grouped_lines[line["character"]].append(line)

            batch_results = {character: [] for character in grouped_lines}

            batch_size = 5  # 设置批次大小
            # 按角色处理
            for character, lines in progress.tqdm(grouped_lines.items(), desc="生成剧本音频"):
                seed = character_seeds.get(character, 0)
                # 按批次处理
                for batch_lines in batch(lines, batch_size):
                    texts = [normalize_zh(line["txt"]) for line in batch_lines]
                    print(f"seed={seed} t={texts} c={character}")
                    wavs = generate_audio_for_seed(chat, int(seed), texts, DEFAULT_BATCH_SIZE, DEFAULT_SPEED,
                                                   refine_text_prompt, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
                                                   DEFAULT_TOP_K, skip_save=True)  # 批量处理文本
                    batch_results[character].extend(wavs)

            # 转换回原排序
            for line in scripts:
                character = line["character"]
                all_wavs.append(batch_results[character].pop(0))

            # 合成所有音频
            audio = combine_audio(all_wavs)
            fname = f"script_{int(time.time())}.wav"
            save_audio(fname, audio)
            return fname


        script_example = {
            "lines": [{
                "txt": "在一个风和日丽的下午，小红帽准备去森林里看望她的奶奶。",
                "character": "旁白"
            }, {
                "txt": "小红帽说",
                "character": "旁白"
            }, {
                "txt": "我要给奶奶带点好吃的。",
                "character": "年轻女性"
            }, {
                "txt": "小红帽，你的篮子里装的是什么？",
                "character": "中年男性"
            }, {
                "txt": "小红帽回答",
                "character": "旁白"
            }, {
                "txt": "这是给奶奶的蛋糕和果酱。",
                "character": "年轻女性"
            }, {
                "txt": "大灰狼心生一计，决定先到奶奶家等待小红帽。",
                "character": "旁白"
            }, {
                "txt": "从此，小红帽再也没有单独进入森林，而是和家人一起去看望奶奶。",
                "character": "旁白"
            }]
        }

        ai_text_default = "武侠小说《天龙八部》 要符合人物背景"

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("### AI脚本")
                gr.Markdown("""
为确保生成效果稳定，仅支持与 GPT-4 相当的模型，推荐使用 4o yi-large deepseek。
如果没有反应，请检查日志中的错误信息。如果提示格式错误，请重试几次。国内模型可能会受到风控影响，建议更换文本内容后再试。

申请渠道（免费额度）：

- [https://platform.deepseek.com/](https://platform.deepseek.com/)
- [https://platform.lingyiwanwu.com/](https://platform.lingyiwanwu.com/)
- [阿里](https://dashscope.console.aliyun.com/apiKey)

                """)
                # 申请渠道

                with gr.Row(equal_height=True):
                    # 选择模型 只有 gpt4o deepseek-chat yi-large 三个选项
                    model_select = gr.Radio(label="选择模型", choices=["gpt-4o", "deepseek-chat", "yi-large", "阿里", "百度", "谷歌"],
                                            value="gpt-4o", interactive=True, )
                with gr.Row(equal_height=True):
                    openai_api_base_input = gr.Textbox(label="OpenAI API Base URL",
                                                       placeholder="请输入API Base URL",
                                                       value=r"https://api.openai.com/v1")
                    openai_api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="请输入API Key",type="password")
                # AI提示词
                ai_text_input = gr.Textbox(label="剧情简介或者一段故事", placeholder="请输入文本...", lines=2,
                                           value=ai_text_default)

                # 生成脚本的按钮
                ai_script_generate_button = gr.Button("AI脚本生成")

            with gr.Column(scale=3):
                gr.Markdown("### 脚本")
                gr.Markdown(
                    "脚本可以手工编写也可以从右侧的AI脚本生成按钮生成。脚本格式 **角色::文本** 一行为一句” 注意是::")
                script_text = "\n".join(
                    [f"{_.get('character', '')}::{_.get('txt', '')}" for _ in script_example['lines']])

                script_text_input = gr.Textbox(label="脚本格式 “角色::文本 一行为一句” 注意是::",
                                               placeholder="请输入文本...",
                                               lines=12, value=script_text)
                script_translate_button = gr.Button("步骤①：提取角色")

            with gr.Column(scale=1):
                gr.Markdown("### 角色种子")
                # DataFrame 来存放转换后的脚本
                # 默认数据
                default_data = [
                    ["旁白", 2222],
                    ["年轻女性", 2],
                    ["中年男性", 2424]
                ]

                script_data = gr.DataFrame(
                    value=default_data,
                    label="角色对应的音色种子，从抽卡那获取",
                    headers=["Character", "Seed"],
                    datatype=["str", "number"],
                    interactive=True,
                    col_count=(2, "fixed"),
                )
                # 生视频按钮
                script_generate_audio = gr.Button("步骤②：生成音频")
        # 输出的脚本音频
        script_audio = gr.Audio(label="AI生成的音频", interactive=False)

        # 脚本相关事件
        # 脚本转换
        script_translate_button.click(
            get_txt_characters,
            inputs=[script_text_input],
            outputs=script_data
        )
        # 处理模型切换
        model_select.change(
            llm_change,
            inputs=[model_select],
            outputs=[openai_api_base_input]
        )
        # AI脚本生成
        ai_script_generate_button.click(
            ai_script_generate,
            inputs=[model_select, openai_api_base_input, openai_api_key_input, ai_text_input],
            outputs=[script_text_input]
        )
        # 音频生成
        script_generate_audio.click(
            generate_script_audio,
            inputs=[script_text_input, script_data],
            outputs=[script_audio]
        )

demo.launch(share=args.share,inbrowser=True)
