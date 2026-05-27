uv sync
uv pip install git+https://github.com/m-bain/whisperx.git

uv run src/a_video_to_audio.py 20260304.140500.0
HF_TOKEN=<YOUR_HF_TOKEN> 
$env:HF_TOKEN="<YOUR_HF_TOKEN>"      
uv run src/b_speaker_diarization.py 20260304.140500.0

# Full

cd /data/disk1/guohaoran/cognitive-coding-engagement-prepare
export HF_HUB_CACHE=/data/disk1/guohaoran/model/huggingface/hub
export HF_TOKEN=<YOUR_HF_TOKEN>
export PATH="$HOME/.local/bin:$PATH"

uv run src/a_video_to_audio.py 20260324.000000.00
uv run src/b_speaker_diarization.py 20260324.000000.00
uv run src/c_automatic_speech_recognition.py 20260324.000000.00
uv run src/d_crowd_detect.py 20260324.000000.00
uv run src/e_extract_frames.py 20260324.000000.00
uv run src/f_image_understanding.py 20260324.000000.00
uv run src/g_merge_results.py 20260324.000000.00

uv run src/batch.py 20260324.000000.00 20260324.000000.01 20260324.000000.02 20260324.000000.03 20260324.000000.04 20260324.000000.05 20260324.000000.06 20260324.000000.07 20260324.000000.08 20260324.000000.09 20260324.000000.10 20260324.000000.11 20260324.000000.12 20260324.000000.13 20260324.000000.14 20260324.000000.15 20260324.000000.16 --task-workers 20
