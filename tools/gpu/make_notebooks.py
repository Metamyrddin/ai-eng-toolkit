"""Generate ready-to-run Kaggle and Google Colab notebooks (and a Kaggle command-line kernel folder).

How this file fits in the project:
    The toolkit's cloud/GPU parts could not be tested on the build machine because it has no GPU. This script
    makes it quick to test them on a FREE GPU. It writes small Jupyter notebooks (.ipynb files) that do five things:
    check the GPU, download (clone) this repository from GitHub, install the packages, run the GPU checks of one
    area (tools/gpu/gpu_checks.py), and save the result file so that you can download it.

What it writes (into tools/gpu/generated/):
    kaggle_area01.ipynb ... kaggle_area08.ipynb, kaggle_all.ipynb   notebooks for Kaggle
    colab_area01.ipynb  ... colab_area08.ipynb,  colab_all.ipynb    notebooks for Google Colab
    kaggle_kernels/<area>/kernel-metadata.json + notebook         folders for "kaggle kernels push"

How to run it:
    python tools/gpu/make_notebooks.py --repo-url https://github.com/Metamyrddin/ai-eng-toolkit.git --kaggle-user YOUR_KAGGLE_NAME
The repository must be public on GitHub, because the notebook clones it without a password.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
AREAS = [f"{i:02d}" for i in range(1, 9)] + ["all"]

UNTESTED_NOTE = ("**Status: UNTESTED on a real GPU.** The code in this notebook was dry-run on a CPU only. "
                 "After you run it on a GPU, please keep the saved JSON file, because it is the proof of the result.")


def md(text):
    """Make a Jupyter "markdown" cell (a cell that holds explanatory text, not code)."""
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    """Make a Jupyter "code" cell (a cell that holds code which runs when you press play)."""
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(cells):
    """Wrap a list of cells into a complete notebook dictionary in the standard nbformat 4 layout."""
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def build(platform, area, repo_url, branch):
    """Build the notebook for one platform ("kaggle" or "colab") and one area ("01".."08" or "all")."""
    title = "all areas" if area == "all" else f"area {area}"
    if platform == "kaggle":
        how = ("1. Open the notebook **Settings** panel and set **Accelerator** to a GPU. "
               "Kaggle only lets you pick a GPU after your phone number is verified.\n"
               "2. In the same panel turn **Internet** on (the notebook has to download code and models).\n"
               "3. Press **Run All**. The results file is saved in the notebook's output files.")
        save = ('import shutil, glob\n'
                'for f in glob.glob("tools/gpu/results/*.json"):\n'
                '    shutil.copy(f, "/kaggle/working/")   # files in /kaggle/working become the notebook output\n'
                'print("Saved to /kaggle/working:", [Path(f).name for f in glob.glob("/kaggle/working/*.json")])')
    else:
        how = ("1. In the menu choose **Runtime**, then **Change runtime type**, and pick a GPU (for example T4).\n"
               "2. Press **Runtime**, then **Run all**.\n"
               "3. The last cell downloads the results file to your computer.")
        save = ('from google.colab import files   # this module only exists inside Colab\n'
                'import glob\n'
                'for f in glob.glob("tools/gpu/results/*.json"):\n'
                '    files.download(f)')
    cells = [
        md(f"# AI engineering toolkit: GPU checks for {title} ({platform.title()})\n\n{UNTESTED_NOTE}\n\n**How to use this notebook**\n\n{how}"),
        code("# Step 1: show which GPU (if any) this machine has.\n!nvidia-smi"),
        code(f'# Step 2: download this repository from GitHub (it must be public).\n'
             f'!git clone --depth 1 --branch {branch} {repo_url} ai-eng-toolkit\n%cd ai-eng-toolkit'),
        code("# Step 3: install the extra packages. PyTorch is already installed on Kaggle and Colab, so we do not touch it.\n"
             "!pip install -q -r requirements-gpu-notebook.txt"),
        code(f'# Step 4: run the small GPU checks. "--device auto" means: use the GPU if there is one.\n'
             f'!python tools/gpu/gpu_checks.py --area {area} --device auto'),
        code("# Step 5: print the saved result file(s) so that you can read them here.\n"
             "import glob, json\nfrom pathlib import Path\n"
             'for f in sorted(glob.glob("tools/gpu/results/gpu_checks_cuda*.json")):\n'
             '    print(f); print(json.dumps(json.load(open(f)), indent=2)[:3000])'),
        code("# Step 6: save the results so that you can keep them.\nfrom pathlib import Path\n" + save),
        md("**Reading the result:** every area shows PASS, FAIL or SKIPPED. The line `is_real_gpu_run: true` inside the JSON file "
           "proves that a real GPU was used. If it says `false`, the notebook ran on the CPU and the result does not count as a GPU test."),
    ]
    return notebook(cells)


def kernel_metadata(user, area):
    """Return the kernel-metadata.json content that `kaggle kernels push` needs (fields from Kaggle's documentation)."""
    slug = f"ai-eng-toolkit-gpu-{'all' if area == 'all' else 'area' + area}"
    return {"id": f"{user}/{slug}", "title": slug.replace("-", " ").title(), "code_file": f"kaggle_{'all' if area == 'all' else 'area' + area}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
            "enable_internet": "true", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}


def main():
    """Write every notebook and every Kaggle kernel folder, then print where they are."""
    ap = argparse.ArgumentParser(description="Generate Kaggle and Colab notebooks that run the toolkit's GPU checks.")
    ap.add_argument("--repo-url", default="https://github.com/Metamyrddin/ai-eng-toolkit.git", help="public GitHub URL of this repository")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--kaggle-user", default="YOUR_KAGGLE_NAME", help="your Kaggle user name (the part after kaggle.com/)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for area in AREAS:
        name = "all" if area == "all" else f"area{area}"
        for platform in ("kaggle", "colab"):
            (OUT / f"{platform}_{name}.ipynb").write_text(json.dumps(build(platform, area, a.repo_url, a.branch), indent=1))
        folder = OUT / "kaggle_kernels" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"kaggle_{name}.ipynb").write_text(json.dumps(build("kaggle", area, a.repo_url, a.branch), indent=1))
        (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(a.kaggle_user, area), indent=2))
    print(f"Wrote {len(AREAS) * 2} notebooks and {len(AREAS)} Kaggle kernel folders into {OUT}")
    if "YOUR_NAME" in a.repo_url:
        print("Reminder: the repository address still contains the placeholder YOUR_NAME, so please pass --repo-url with the real address.")


if __name__ == "__main__":
    main()
