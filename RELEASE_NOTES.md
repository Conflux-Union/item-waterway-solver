# Release Notes

## v0.1.0

Initial public release of the Rust item-waterway search tool.

### Highlights

- Added a Rust CLI for searching slime-piston-launched Minecraft 1.17.1 item waterways.
- Ported the search flow from the earlier JavaScript prototype to a release-friendly binary.
- Corrected fluid handling order so current is applied once after movement instead of being injected before and after the movement step.
- Replaced the ad hoc underwater threshold with the Minecraft item entity `> 0.1` fluid height rule.
- Added CSV, Markdown, and JSON reporting for ranked candidates.
- Added tests that lock the corrected fluid tracker and movement ordering behavior.

### Release Asset

This release includes a Linux release binary package for `item-waterway-solver`.
