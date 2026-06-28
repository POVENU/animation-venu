#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
SHARED_DIR = OUTPUT_DIR / "_shared"
IMAGES_DIR = SHARED_DIR / "images"
TEMP_DIR = OUTPUT_DIR / "_temp"

VIDEO_PRESETS = {
    "short": {
        "target_duration_sec": 30,
        "scene_count": 6,
        "words": 90,
    },
    "medium": {
        "target_duration_sec": 90,
        "scene_count": 12,
        "words": 220,
    },
    "long": {
        "target_duration_sec": 180,
        "scene_count": 20,
        "words": 450,
    },
}


def ensure_dirs():
    for directory in [OUTPUT_DIR, SHARED_DIR, IMAGES_DIR, TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_\\-]", "", value)
    return value or "language"


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def get_video_dimensions(aspect_ratio: str):
    if aspect_ratio == "9:16":
        return 1080, 1920
    if aspect_ratio == "1:1":
        return 1080, 1080
    return 1920, 1080


def strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if len(lines) >= 2:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    return text


def create_master_storyboard(client, prompt: str, video_type: str, style: str):
    preset = VIDEO_PRESETS[video_type]

    system = "You are an expert Pixar animation movie writer. Return ONLY valid JSON."

    user = f"""
Create a cinematic Pixar-style animated storyboard.

Story:
"{prompt}"

Requirements:

- Video type: {video_type}
- Duration: {preset["target_duration_sec"]} seconds
- Scene count: {preset["scene_count"]}
- Style: {style}

Return JSON format:

{{
  "title": "string",
  "story_summary": "string",
  "characters": [
    {{
      "name": "string",
      "appearance": "string"
    }}
  ],
  "scenes": [
    {{
      "scene_number": 1,
      "title": "string",
      "emotion": "happy",
      "camera_motion": "slow zoom in",
      "character_action": "rabbit jumping happily",
      "narration_english": "string",
      "image_prompt": "string"
    }}
  ]
}}

Rules:

- Characters must stay visually consistent.
- Include strong facial expressions.
- Include body movement.
- Include cinematic lighting.
- Include dynamic camera angles.
- Use emotional storytelling.
- Use Pixar/DreamWorks quality visuals.
- Image prompts must describe animated movie scenes only.

Return ONLY JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.8,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = strip_code_fences(response.choices[0].message.content)
    return json.loads(raw)


def generate_image_with_openai(client, prompt, out_path: Path, width: int, height: int):
    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )

        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        out_path.write_bytes(image_bytes)

        img = Image.open(out_path).convert("RGB")
        img = img.resize((width, height))
        img.save(out_path)

    except Exception as e:
        print("Image generation failed:", e)
        create_fallback_image(prompt, out_path, width, height)


def create_fallback_image(text_prompt, out_path: Path, width: int, height: int):
    img = Image.new("RGB", (width, height), color=(20, 20, 40))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            40,
        )
    except Exception:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(text_prompt[:500], width=40)
    draw.multiline_text((60, 100), wrapped, fill=(255, 255, 255), font=font, spacing=8)
    img.save(out_path)


def generate_animation_frames(client, base_prompt, scene_dir: Path, width: int, height: int):
    prompts = [
        f"{base_prompt}, first frame, character starting movement",
        f"{base_prompt}, middle frame, character moving dynamically",
        f"{base_prompt}, final frame, emotional cinematic pose",
    ]

    frame_paths = []

    for idx, prompt in enumerate(prompts):
        frame_path = scene_dir / f"frame_{idx:02}.png"
        generate_image_with_openai(client, prompt, frame_path, width, height)
        frame_paths.append(frame_path)

    return frame_paths


def translate_text(client, text: str, target_language: str = "Telugu") -> str:
    """
    Translate given English text into the target language.
    Returns the translated text only (no fences).
    """
    system = "You are a professional translator. Translate the user's English text into the target language, preserving tone, meaning, and punctuation. Return only the translated text with no extra commentary."
    user = f"Translate the following text to {target_language}. Keep lines short and suitable for voiceover (try to preserve pauses with punctuation):\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    translated = strip_code_fences(response.choices[0].message.content).strip()
    return translated


def generate_tts(client, text: str, out_path: Path, voice: Optional[str] = "alloy"):
    """
    Uses OpenAI TTS to synthesize `text` and stream to out_path file.
    """
    if not text:
        # create a silent short audio file as fallback
        with open(out_path, "wb") as f:
            f.write(b"")
        return

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    ) as response:
        response.stream_to_file(out_path)


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def mix_with_background(narration_path: Path, background_path: Path, out_path: Path, bg_volume: float = 0.15):
    """
    Mix narration with looped background music, lowering music volume so narration is clear.
    """
    # If background doesn't exist, copy narration through
    if not background_path.exists():
        shutil.copy(narration_path, out_path)
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(narration_path),
        "-stream_loop",
        "-1",
        "-i",
        str(background_path),
        "-filter_complex",
        f"[1:a]volume={bg_volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2,volume=1.0",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_path),
    ]

    run_cmd(cmd)


def create_scene_video(frame_paths: List[Path], audio_path: Path, out_path: Path, width: int, height: int, fade_duration: float = 0.5):
    """
    Create a cinematic scene video from frames and narration audio.
    Adds zoompan and minterpolate for smoother motion. Applies fade in/out.
    Returns the duration (seconds) of the scene (audio).
    """
    duration = get_audio_duration(audio_path)

    temp_video = TEMP_DIR / f"{out_path.stem}_temp.mp4"

    # pattern expects frame_00.png, frame_01.png, frame_02.png
    input_pattern = str(frame_paths[0]).replace("00", "%02d")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "1",
        "-i",
        input_pattern,
        "-vf",
        (
            f"fps=30,"
            f"scale={width}:{height},"
            f"zoompan=z='min(zoom+0.0015,1.15)':"
            f"d=125:s={width}x{height},"
            f"minterpolate='fps=30'"
        ),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(temp_video),
    ]

    run_cmd(cmd)

    # merge video and narration audio
    merged_video = TEMP_DIR / f"{out_path.stem}_merged.mp4"
    cmd2 = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(merged_video),
    ]

    run_cmd(cmd2)

    # apply fade in/out to the merged video and write to out_path
    fade_out_start = max(0, duration - fade_duration)
    faded = TEMP_DIR / f"{out_path.stem}_faded.mp4"
    vf_filter = f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_out_start}:d={fade_duration}"
    cmd3 = [
        "ffmpeg",
        "-y",
        "-i",
        str(merged_video),
        "-vf",
        vf_filter,
        "-c:a",
        "copy",
        str(faded),
    ]

    run_cmd(cmd3)

    # move faded to final out_path
    shutil.move(str(faded), str(out_path))

    # clean up temps (keep temp_video if needed)
    for p in [temp_video, merged_video]:
        try:
            p.unlink()
        except Exception:
            pass

    return duration


def concatenate_videos(video_files: List[Path], output_file: Path):
    """
    Concatenate processed scene videos using ffmpeg concat demuxer.
    """
    concat_file = TEMP_DIR / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for file in video_files:
            f.write(f"file '{file.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_file),
    ]

    run_cmd(cmd)


def write_srt(scenes: List[dict], audio_files: List[Path], out_srt: Path):
    """
    Create a simple SRT with each scene's narration. Uses audio durations to set timing.
    """
    def format_timestamp(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    current = 0.0
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, (scene, audio_path) in enumerate(zip(scenes, audio_files), start=1):
            duration = get_audio_duration(audio_path)
            start = current
            end = current + duration
            text = scene.get("narration_telugu") or scene.get("narration_english") or ""
            text = text.strip()
            # Limit line length to reasonable subtitle lengths
            if len(text) > 200:
                text = text[:197] + "..."
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(text + "\n\n")
            current = end


def build_shared_images(client, scenes, width, height, style):
    for idx, scene in enumerate(scenes, start=1):
        scene_dir = IMAGES_DIR / f"scene_{idx:02}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        prompt = (
            f"{scene['image_prompt']}. "
            f"Emotion: {scene.get('emotion', 'happy')}. "
            f"Action: {scene.get('character_action', '')}. "
            f"Camera motion: {scene.get('camera_motion', '')}. "
            f"Style: {style}. "
            f"Pixar style 3D animation. "
            f"DreamWorks cinematic lighting. "
            f"Expressive cartoon characters. "
            f"Dynamic movement pose."
        )
        generate_animation_frames(client, prompt, scene_dir, width, height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--video-type", required=True)
    parser.add_argument("--languages", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--aspect-ratio", required=True)
    parser.add_argument("--background-music", required=False, help="Optional path to background music (mp3). If not set, will look for background.mp3 in repo root or skip.")
    parser.add_argument("--fade-duration", required=False, type=float, default=0.5, help="Fade in/out seconds for each scene.")
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    ensure_dirs()

    width, height = get_video_dimensions(args.aspect_ratio)

    storyboard = create_master_storyboard(client, args.prompt, args.video_type, args.style)

    # Build shared frames
    build_shared_images(client, storyboard["scenes"], width, height, args.style)

    scene_videos = []
    scene_audio_files = []

    is_telugu = args.languages.strip().lower().startswith("telu")
    # background music: flag / env var / file
    bg_music = None
    if args.background_music:
        bg_music = Path(args.background_music)
    elif os.getenv("BACKGROUND_MUSIC"):
        bg_music = Path(os.getenv("BACKGROUND_MUSIC"))
    else:
        candidate = ROOT / "background.mp3"
        if candidate.exists():
            bg_music = candidate

    for idx, scene in enumerate(storyboard["scenes"], start=1):
        scene_dir = IMAGES_DIR / f"scene_{idx:02}"
        frame_paths = sorted(scene_dir.glob("frame_*.png"))

        audio_file = TEMP_DIR / f"scene_{idx:02}.mp3"
        audio_file_mixed = TEMP_DIR / f"scene_{idx:02}_mixed.mp3"
        video_file = TEMP_DIR / f"scene_{idx:02}.mp4"
        processed_video = TEMP_DIR / f"scene_{idx:02}_processed.mp4"

        # pick narration
        narration_text = scene.get("narration_english", "").strip()
        if is_telugu and narration_text:
            try:
                telugu_text = translate_text(client, narration_text, "Telugu")
            except Exception as e:
                print("Translation failed, using English narration. Error:", e)
                telugu_text = narration_text
            scene["narration_telugu"] = telugu_text
            tts_input = telugu_text
        else:
            tts_input = narration_text

        # generate TTS into audio_file
        print(f"Generating TTS for scene {idx}...")
        generate_tts(client, tts_input, audio_file)

        # mix with background if available
        if bg_music:
            print(f"Mixing scene {idx} narration with background music {bg_music}...")
            try:
                mix_with_background(audio_file, bg_music, audio_file_mixed, bg_volume=0.12)
                audio_to_use = audio_file_mixed
            except Exception as e:
                print("Background mixing failed, using narration only:", e)
                audio_to_use = audio_file
        else:
            audio_to_use = audio_file

        # create scene video (with fades)
        print(f"Creating scene video for scene {idx}...")
        duration = create_scene_video(frame_paths, audio_to_use, processed_video, width, height, fade_duration=args.fade_duration)

        # move processed to a stable video file list for concatenation
        final_scene_video = OUTPUT_DIR / f"scene_{idx:02}.mp4"
        shutil.copy(processed_video, final_scene_video)
        scene_videos.append(final_scene_video)
        scene_audio_files.append(audio_to_use)

    # write SRT captions (Telugu if present)
    if is_telugu:
        srt_path = OUTPUT_DIR / "final_video_telugu.srt"
        write_srt(storyboard["scenes"], scene_audio_files, srt_path)
        print("Wrote subtitles to", srt_path)

    # concatenate final scene videos
    final_video = OUTPUT_DIR / "final_video.mp4"
    concatenate_videos(scene_videos, final_video)

    print("Final cinematic animation video created at:", final_video)
    if is_telugu:
        print("Telugu subtitles file (SRT):", OUTPUT_DIR / "final_video_telugu.srt")


if __name__ == "__main__":
    main()
