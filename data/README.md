# Data

| File             | Contents                                                                   |
| ---------------- | -------------------------------------------------------------------------- |
| `train_100k.npz` | training data, 100,000 labeled samples (413 MB)                            |
| `test_demo.npz`  | demo test set, 400 labeled samples drawn like the evaluation data (1.7 MB) |

The training data is too large for git, so both files are published as
[release](https://github.com/GDG-UAM/ml-contest-materials/releases/latest) files. Download them into
this folder, from the repository folder:

```bash
# macOS / Linux / Git Bash
curl -L -o data/train_100k.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/train_100k.npz
curl -L -o data/test_demo.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/test_demo.npz
```

```powershell
# Windows PowerShell (curl.exe, not the curl alias)
curl.exe -L -o data/train_100k.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/train_100k.npz
curl.exe -L -o data/test_demo.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/test_demo.npz
```

Or open the release page in your browser and save both files here.

Check the downloads (optional):

```bash
sha256sum data/train_100k.npz data/test_demo.npz # macOS: shasum -a 256 data/train_100k.npz data/test_demo.npz
# 1c231e9a807c348b34290415f56e7c7b99543a81673a9d3cef1002b1a42e5668  data/train_100k.npz
# 03fce51b3dacfaaa04418c4907def59874c4b6963c0561c9ccac83bbca90aac0  data/test_demo.npz
```

```powershell
Get-FileHash data/train_100k.npz, data/test_demo.npz   # SHA256, same values in upper case
```

Load them with `competition_api.load_dataset("data/train_100k.npz")`; see the main README.
