# Dependency pinning

The [Scorecard report](https://api.securityscorecards.dev/projects/github.com/open-edge-platform/anomalib)
from 2026-10-23 scanned commit `0be85508edde53a2b1062ee6d1732c85c5b7172e`
and scored Pinned-Dependencies 8/10. It reported 12 workflow references,
12 pip commands, and one installer download.

## Workflow references

The 12 flagged references in `pr.yaml`, `release-lib.yaml`, and
`security-checks.yaml` used the invalid prefix `$/.github/workflows/`.
They now use `./.github/workflows/`, so GitHub runs the reusable workflow
from the same commit as the caller. Remote actions already use full commit hashes.
Keep those hashes when updating actions.

## CI tools

`build.txt`, `twine.txt`, and `commitizen.txt` pin the existing tool versions
and their transitive dependencies with SHA-256 hashes. The artifact builder,
PR title check, and version bump workflows install them with `--require-hashes`.
These cover the six flagged CI tool installation commands.

To update a tool, edit its `.in` file, then regenerate the matching `.txt` file
from the repository root (replace `build` with `twine` or `commitizen`):

```bash
uv --no-config pip compile .github/requirements/build.in --generate-hashes --universal --python-version 3.10 --no-annotate -o .github/requirements/build.txt
```

`--no-config` prevents application-specific uv overrides from affecting CI tools.
Review the versions and hashes together. Validate installation on the workflow's
Python version before merging. These files lock tools, not the package's isolated
build environment or its runtime dependencies.

## Installer

`examples/cli/00_installation/uv_install.sh` downloads the uv 0.9.18 installer
from its versioned GitHub release and verifies its SHA-256 before execution.
This matches the uv version selected in `.github/renovate.json5`.
Download failure or checksum mismatch stops the example before any installation.
Update the URL and checksum together after reviewing the upstream release.

The upstream installer selects platform binaries from that same versioned
release. It does not embed binary checksums. This is a version-pinning exception
for those second-stage downloads: retain upstream platform selection and install
behavior rather than maintaining a separate platform installer here. Release
asset replacement remains a risk; revisit when upstream embeds binary checksums.

## Intentional pip exceptions

These six original findings remain visible in Scorecard; they are not suppressed:

- The five commands in `examples/cli/00_installation/pip_install.sh` demonstrate
  installing the latest Anomalib release, choosing hardware extras, and installing
  a development checkout. Fixed hashes would change that documented behavior and
  require separate locks for each supported platform and backend. Keep these
  examples flexible; use the committed `uv.lock` for reproducible development.
- `download-url-check.yaml` installs the current checkout with `[cpu,test]` to
  test dataset URLs. Pip hash checking cannot cover an editable local project.
  This scheduled health check intentionally exercises the project's declared
  dependency ranges; it does not build or publish release artifacts.

Both exceptions allow dependency resolution to change over time. Revisit them
if these paths become release inputs or promise reproducible installations.
An overall score of 10 is not expected while these documented exceptions remain.
