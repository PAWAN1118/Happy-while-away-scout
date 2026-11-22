# SniffStation Scout

SniffStation Scout is an early prototype for a dog enrichment companion. The idea is to pair a physical scent and food activity with a friendly on-screen character that can notice a few visible changes and respond while a person is away.

The project is still in progress. The current build demonstrates a website, a local camera preview, a small set of camera-based estimates, and a scripted scent-hunt interaction. It is a concept prototype, not a finished pet monitor or a validated dog-behavior product.

## The idea

Dogs can benefit from safe, varied enrichment when their people are busy or away. SniffStation is envisioned as a product ecosystem with:

- A physical scent-based enrichment activity designed around a dog's natural sniffing and food-searching behavior.
- Scout, a friendly dog character in the website that gives brief spoken guidance.
- A camera-aware experience that can react to simple visible activity, such as a dog moving or being near a configured feeder area.
- An owner-facing experience that could eventually show what happened during an enrichment session.

The current software only prototypes parts of this idea. It does not include the physical feeder, a production monitoring service, or a trained dog-behavior understanding model.

## What works in this prototype

- A responsive SniffStation website with the existing Scout character and scent-hunt concept.
- An optional browser camera preview.
- A FastAPI service in `server.py` that receives individual camera snapshots for dog detection. In a Vercel deployment, those snapshots are processed by Vercel.
- Pretrained YOLO11 nano object detection for locating a dog in a frame.
- Simple estimates based on dog position over time: `moving`, `still`, `near_feeder`, or `not_visible`.
- Short, preset Scout replies when the estimate changes.
- Optional spoken replies using the browser's built-in text-to-speech.
- Simulated scent-hunt controls in the page for demonstrating the product flow without a dog or camera.

## What it does not do yet

- It does not understand dog language, translate barking, or infer what a dog is thinking or feeling.
- It does not reliably identify eating, sniffing, pacing, anxiety, distress, or other detailed behaviors.
- `near_feeder` means the detected dog is near a configured region of the image. It does not confirm that food was eaten or that the dog interacted with a device.
- Scout's replies are preset text. This is not a live conversational AI agent.
- It does not record video, save a behavior timeline, send alerts, or run as a background service.
- Camera analysis runs only while the page is open and the camera preview is active. The local version sends frames to localhost; the Vercel version sends them to Vercel.
- This prototype has not been evaluated for accuracy, animal welfare, or unattended use. Do not rely on it for safety or health monitoring.

## Project files

| File | Purpose |
| --- | --- |
| `index.html` | Website, Scout interface, local camera preview, simulated scent hunt, and browser-side frame sending. |
| `server.py` | FastAPI server, YOLO11 dog detection, simple activity estimates, and Scout response text. |
| `requirements.txt` | Python packages needed by the local server. |
| `vercel.json` | Vercel Python function configuration and deployment file selection. |
| `train.py` | Experimental dog-pose fine-tuning script. It is not required to run the current prototype, and the current `server.py` does not load its output. |
| `init.py` | Separate hackathon environment/setup script; it is not needed to run SniffStation locally. |

## Run locally

Use Python 3.10 or later. Open PowerShell in this project folder.

Install the packages:

```powershell
python -m pip install -r requirements.txt
```

Start the local site and API:

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in a browser. The first server start may download the pretrained `yolo11n.pt` weights. Check [http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status) for model readiness.

To try the camera experience:

1. Open Scout Live and start the camera preview.
2. Allow camera access in the browser.
3. Keep the dog visible in the frame and wait a few seconds for repeated detections.
4. Enable Scout's voice option if you want the browser to read replies aloud.
5. Stop the camera when you are done.

The simulated scent-hunt controls can be used separately to demonstrate the interaction without a dog. For a camera test without a dog, a clear dog photo or video shown on another screen may help check whether the detector sees a dog, but this is only a pipeline check and does not validate real behavior recognition.

## Camera and privacy notes

When run locally, the browser sends sampled image frames to the Python server on the same computer. In the Vercel deployment, the browser sends snapshots to the hosted Python service for analysis. The application code analyzes frames in memory and does not intentionally save them. The camera and analysis stop when monitoring is stopped or the page is closed.

The prototype has no login or owner authentication. Do not use it for private camera feeds, safety monitoring, or unattended care. It is a public concept demo, not a production monitoring service.

## Current technical approach

The active server loads `yolo11n.pt`, a pretrained object detector. It filters detections to the COCO `dog` class and returns the dog's image position. The browser estimates movement and feeder proximity. The feeder area is currently a fixed image region in `index.html` (`feeder` coordinates); change those values to match the camera view.

The included `train.py` is an experiment that fine-tunes a dog-pose model for 60 epochs. It is not on the active inference path: the running server uses YOLO11 object detection and does not read the pose-training checkpoint. Running `train.py` is unnecessary for the current app and can take a long time.

## Roadmap

- Preserve and refine the current SniffStation design while integrating camera status and Scout replies cleanly.
- Add a way to select or calibrate the feeder area instead of editing image coordinates in Python.
- Improve dog tracking so movement estimates are less sensitive to camera noise and temporary missed detections.
- Build a properly labeled, dog-specific video dataset before attempting to classify actions such as eating, sniffing, resting, and pacing.
- Test with different dogs, camera placements, lighting, and environments; show uncertainty when the model cannot tell.
- Add owner-controlled privacy settings and an event history only after the analysis is reliable and storage behavior is clearly defined.
- Evaluate all spoken prompts for calm, safe use around dogs.

## Project status

**Status: incomplete prototype / concept demonstration.** The project explores how a camera-aware character could make a dog enrichment product feel more responsive. The next major work is validating the behavior estimates and designing the physical enrichment product around how dogs actually interact with scent and food activities.
