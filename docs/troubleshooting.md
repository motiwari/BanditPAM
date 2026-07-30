# Troubleshooting

## `pip install banditpam` fails on Python 3.13+ ("Failed to build wheel")

**Symptom:** Installing via `pip install banditpam` tries to compile from
source and fails, often with errors related to `setuptools`, missing build
tools, or a `build-essential` warning on Linux.

**Cause:** BanditPAM's published PyPI wheels currently target Python 3.8–3.12.
On systems where Python 3.13 or newer is the default (e.g. recent Ubuntu
releases ship Python 3.14), pip cannot find a matching prebuilt wheel and
falls back to a source build, which requires a full C++ toolchain and
Armadillo already installed.

**Fix:** Install a supported Python version alongside your system's default
and create your virtual environment with it.

On Ubuntu, if `python3.11` isn't available via `apt` by default, add the
deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv build-essential
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install banditpam
```

This should download a prebuilt `manylinux` wheel in seconds rather than
compiling.

## Installing on Windows

There are currently no prebuilt Windows wheels on PyPI, and building from
source on Windows requires a full MSVC + Armadillo + OpenMP setup (see
`docs/install_windows.md`). For most users, the fastest path is to use
**WSL (Windows Subsystem for Linux)** and follow the Linux installation
instructions, which have full wheel support:

```powershell
wsl --install
```

Then, inside the Ubuntu shell, follow the steps above.

## `test_smaller.py` / `test_larger.py` fail with `FileNotFoundError: data/MNIST_70k.csv`

Some tests require a larger MNIST dataset (`MNIST_70k.csv`) that is not
committed to the repository (only the smaller `data/MNIST_1k.csv` is
included). These specific tests will fail with a `FileNotFoundError` unless
you supply this file yourself. This does not indicate a problem with your
installation — the smaller, bundled-data tests will still pass.