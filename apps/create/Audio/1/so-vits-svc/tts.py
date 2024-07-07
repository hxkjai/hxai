import asyncio
import random
import edge_tts
from edge_tts import VoicesManager
import sys
from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0

TEXT = sys.argv[1]
LANG = detect(TEXT) if sys.argv[2] == "Auto" else sys.argv[2]
if LANG == "zh-cn" or LANG == "zh-tw":
    LOCALE = LANG[:-2] + LANG[-2:].upper()
RATE = sys.argv[3]
VOLUME = sys.argv[4]
GENDER = sys.argv[5] if len(sys.argv) == 6 else None
OUTPUT_FILE = "tts.wav"

print("Running TTS...")
print(f"Text: {TEXT}, Language: {LANG}, Gender: {GENDER}, Rate: {RATE}, Volume: {VOLUME}")

async def _main() -> None:
    voices = await VoicesManager.create()
    if not GENDER is None:
        if LANG.startswith("zh"):
            voice = voices.find(Gender=GENDER, Locale=LOCALE)
        else:
            voice = voices.find(Gender=GENDER, Language=LANG)
        VOICE = random.choice(voice)["Name"]
    else:
        VOICE = LANG
    communicate = edge_tts.Communicate(text = TEXT, voice = VOICE, rate = RATE, volume = VOLUME)
    await communicate.save(OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_main())