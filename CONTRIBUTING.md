# Contributing

Thanks for your interest in improving DoomScroller600. Contributions may include bug fixes, accessibility improvements, documentation, tests, or thoughtful changes to the gesture logic.

## Before making a change

1. Search existing issues and pull requests for the same problem or proposal.
2. For a larger behavior change, open an issue first so the expected behavior can be discussed.
3. Fork the public repository and create a focused branch for your change.

## Local setup

Use Python 3.10 or newer and follow the setup steps in [README.md](README.md). The current supplied source has no `--dry-run` option; take care to focus a safe window while working on gesture behavior.

Before opening a pull request, check Python syntax with:

```powershell
python -m compileall -q DoomScroller600.py
```

The program opens the default camera and can scroll the active window when started; it has no dry-run mode. In your pull request, say which checks you ran and which you could not run. Do not upload webcam recordings or identifiable face images to issues or pull requests.

## Pull requests

- Keep each pull request focused and explain the user-visible change.
- Include steps to reproduce a bug or verify a fix.
- Update README, privacy notes, third-party notices, and the changelog when behavior or data handling changes.
- Preserve existing copyright and license notices. Under MIT, include the license notice when redistributing substantial portions of the code.
- A repository link is requested when substantial code is reused, as a courtesy; it is not an additional condition of the MIT License.
- Be respectful and follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Submitting a contribution does not transfer copyright. Contributors retain rights to their contributions and agree to license contributions under the repository's MIT License when submitting them.

For hiring inquiries, questions, suggestions, contribution ideas, or other messages, [DM the maintainer on Discord](https://discord.com/users/543177724450308136). Please do not send private camera recordings or credentials.
