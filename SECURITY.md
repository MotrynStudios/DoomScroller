# Security Policy

## Reporting a vulnerability

Please do not post security-sensitive details, private camera footage, credentials, or personal information in a public issue. If GitHub's private vulnerability reporting is enabled for this repository, use the **Report a vulnerability** option on the Security tab. Otherwise, [DM the maintainer on Discord](https://discord.com/users/543177724450308136) to arrange a private report.

## Scope

The app reads webcam frames for local face landmark processing and sends scroll input to the active desktop application. This supplied source has no hand tracking and does not download model files. It is not intended to identify people or store camera recordings.
