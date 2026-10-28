# Training data

`train_100k.npz` (413 MB, 100,000 labeled samples) is too large for git, so it is published as a
[release](https://github.com/GDG-UAM/ml-contest-materials/releases/latest) file. Download it into
this folder, from the repository folder:

```bash
# macOS / Linux / Git Bash
curl -L -o data/train_100k.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/train_100k.npz
```

```powershell
# Windows PowerShell (curl.exe, not the curl alias)
curl.exe -L -o data/train_100k.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/train_100k.npz
```

Or open the release page in your browser and save `train_100k.npz` here.

Check the download (optional):

```bash
sha256sum data/train_100k.npz                    # macOS: shasum -a 256 data/train_100k.npz
# 1c231e9a807c348b34290415f56e7c7b99543a81673a9d3cef1002b1a42e5668
```

```powershell
Get-FileHash data/train_100k.npz                 # SHA256, same value in upper case
```

Load it with `competition_api.load_dataset("data/train_100k.npz")`; see the main README.
