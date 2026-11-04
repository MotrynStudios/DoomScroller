# DoomScroller600

**Version 0.5P — in development**

A local webcam prototype that scrolls the active window when the user raises their eyebrows. This clean-start release uses the supplied `Scroller1200.py` program as its baseline and keeps the project filename `DoomScroller600.py`.

## Gestures and current limitations

- Raising both eyebrows scrolls forward; raising one eyebrow scrolls backward.
- A raised-brow gesture must last at least 0.08 seconds. Gestures held longer than 0.75 seconds are discarded.
- This source has **no hand gesture detection**.
- Automatic recalibration is **off by default**. Press the apostrophe key to toggle it on or off. Press `C` to restart calibration manually.
- The app can label the head as facing away based on its head-orientation estimate. It does not use “a part of the face is not visible” as the only away condition.
- Expression guards and landmark estimates are experimental. They can suppress intended gestures or miss expressions; they do not identify emotions reliably.

## Setup

Use Python 3.10 or newer. From the project folder in PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python DoomScroller600.py
```

The script opens the default camera and a resizable preview window. Keep the intended browser or app active because scrolling is sent to the currently focused window. Press `Q` to quit. Press `C` to recalibrate; press the apostrophe key to toggle automatic recalibration; press backslash to set a manual threshold; press comma to hide or show overlay details.

## Privacy and safety

Camera frames are processed locally by OpenCV and MediaPipe Face Mesh. This script does not include a camera upload or recording feature. PyAutoGUI sends scroll actions to the active window. Its failsafe is disabled in this supplied source, so use it carefully and close the program with `Q` when finished.

This is an experimental assistive prototype, not a medical device or a guarantee of accessibility. Begin in a safe window and confirm that the controls behave as expected before using it with important content.

See [Privacy](PRIVACY.md), [Terms of Use](TERMS_OF_USE.md), [Third-Party Notices](THIRD_PARTY_NOTICES.md), and [Security](SECURITY.md). Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). The project is available under the [MIT License](LICENSE). Retain the copyright and license notice when redistributing.

## Contact

For hiring inquiries, questions, suggestions, collaboration, or other messages, [DM me on Discord](https://discord.com/users/543177724450308136). DMs are open. Please do not send passwords or private camera recordings.
