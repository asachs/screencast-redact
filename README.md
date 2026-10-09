# screencast-redact

Find a piece of text, such as an email address, a hostname or a name, everywhere it appears in a long screen recording, and cover it with black boxes. Nothing else in the recording is touched.

Built in a hurry on 9 October 2026 to strip an email address out of a two-hour, 4K incident-response screencast before submission. It took a shell prompt that appeared in hundreds of places and a browser tab title, and produced a clean file in about a quarter of an hour on a laptop.

## How it works

Screen recordings render the same text the same way every time: same font, same size, same colours. So a small PNG crop of the rendered text is a reliable template, and OpenCV's normalised template matching finds every occurrence with a clear margin between real hits (scores around 0.8 to 0.95) and unrelated text (below 0.6). The scan runs on a small low-frame-rate proxy, in parallel across cores. Each detection becomes a timed black box, padded a little in time and space, and the boxes are burned into the full-resolution original by ffmpeg, again in parallel by time segment, then joined without re-encoding.

## Install

ffmpeg and ffprobe on the PATH, Python 3.9 or later, and:

```
pip install -r requirements.txt
```

## Use

1. Grab a few frames where the text is visible and crop the text out of them at full resolution, one PNG per rendering (the terminal prompt and a browser tab title are two different renderings). Any image editor works; keep a little background around the letters.

```
python3 redact.py frames recording.mov 60 420 1800 --out frames
```

2. Make a proxy and scan it. Start with `--calibrate` to see the scores; real hits and false matches should be separated by a wide gap, and the threshold goes in that gap.

```
python3 redact.py proxy recording.mov proxy.mp4
python3 redact.py scan recording.mov proxy.mp4 templates/*.png --partial --calibrate
```

`--partial` also matches the left and right halves of each template, which catches text cut off at a window edge. `--band y0:y1` restricts the search to a horizontal band in full-resolution pixels, which is fast for text that lives in a fixed bar such as a browser tab strip.

3. Render, then look at the verification crops before you ship anything.

```
python3 redact.py render recording.mov detections.json --out redacted.mov
python3 redact.py verify redacted.mov detections.json --out verify
```

## Things learned the hard way

- `ffmpeg -ss ... -copyts` keeps the original timestamps so `enable='between(t,a,b)'` windows stay absolute across segments. With `-copyts` the end of a segment must be given with `-to` (absolute); `-t` is compared against the shifted timestamps and the segment ends immediately.
- Apple's hardware encoder is a single shared device. Eight parallel `h264_videotoolbox` jobs queue on it and the CPU sits idle. `libx264 -preset ultrafast` with two threads per job uses every core and was several times faster for this workload.
- A long MOV with a fine timebase overflows the 32-bit duration field and QuickTime refuses to play it. The join step sets `-video_track_timescale 15360`.
- A box, not a blur. Blurred text at known font and size can be read back; a box cannot.
- Verify from the rendered file, not from the detections. Crop frames from inside the covered spans and look at them.

## Limits

Template matching finds the rendering you gave it. The same string in a different font, size or colour, for example inside an email body rather than the prompt, needs its own template. Text that scales with window zoom needs one template per scale. Nothing here reads text; it only matches pixels.

## Licence

MIT. Written by André Sachs with editorial assistance from Claude (Anthropic).
