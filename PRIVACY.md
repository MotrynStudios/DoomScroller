# Privacy Notice

**Effective date:** 2026-11-04

DoomScroller600 is a local desktop prototype maintained by Mohamed Ayman Elnashar. This notice describes the behavior of the source code in this repository. It does not describe GitHub, your operating system, Python package providers, camera drivers, or the applications you control.

## Camera processing

When the program runs, it reads frames from the default camera and processes facial landmarks locally with OpenCV and MediaPipe Face Mesh. The supplied source does not include a camera recording, upload, account, analytics, or cloud-processing feature. Landmark measurements are held in memory while the program runs and are not saved by this source.

The program sends scroll input to the currently focused desktop application through PyAutoGUI. Its failsafe is disabled in this supplied source. Close the program with `Q` when you are finished and take care to keep the intended application focused.

## Network and files

The supplied program does not download model files or send webcam frames to the project maintainer. Python dependencies are installed separately from their package sources. Your operating system, camera driver, installed packages, and the application receiving scroll input may have separate data practices.

## Your choices

You choose when to run the camera and can stop the program with `Q` or `Ctrl+C`. Do not use the camera to observe other people without considering their privacy and consent. Please do not include identifiable face images or camera recordings in public issues.

For privacy questions, contact the maintainer through [the project repository](https://github.com/theinevitableinvincible/DoomScroller600) or [Discord](https://discord.com/users/543177724450308136).
