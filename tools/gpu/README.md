# Free GPU runner: test the cloud parts of this toolkit without paying

This folder helps you run the toolkit's GPU checks on a free cloud GPU from Kaggle or from Google Colab.
Nothing in this folder has been run on a real GPU yet, because the computer that built the toolkit has no GPU.
Every GPU result is therefore marked **UNTESTED** until you run it and keep the result file.
The same code was run on a normal CPU as a "dry run", and all eight area checks passed there.

## What is in this folder

- `gpu_checks.py` is the file that does the real work. It has one small test for each of the eight areas, and it can run on a GPU or on a CPU.
- `make_notebooks.py` writes ready-to-run notebooks for Kaggle and for Colab into a folder called `generated/`.
- `kaggle_run.sh` is a one-command way to run a check on Kaggle from your own computer, using the Kaggle command line.
- `results/` is where result files are saved. Each result file says which device was really used, so a CPU dry run can never be mistaken for a GPU run.

## Before you start (one time only)

1. Put this repository on GitHub and make it **public**. The notebooks download the code with `git clone`, and they cannot type a password for you.
2. Create the notebooks with your own details. Run this on your computer inside the project's virtual environment:

   ```bash
   python tools/gpu/make_notebooks.py --repo-url https://github.com/Metamyrddin/ai-eng-toolkit.git --kaggle-user YOUR_KAGGLE_NAME
   ```

   The notebooks appear in `tools/gpu/generated/`.

## Path A: the easiest way, using the Kaggle website

Kaggle gives every verified user a free GPU for a limited number of hours each week.
Kaggle's own documentation says the weekly quota is 30 hours "or sometimes higher depending on demand and resources" and that one session can run for at most 12 hours
([Kaggle: efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage), [Kaggle: notebooks documentation](https://www.kaggle.com/docs/notebooks)).
The free GPU type changes over time, so look at the accelerator menu to see what you get.

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Verify your phone number in your Kaggle account settings. Kaggle asks for this before it lets you switch on a GPU or the internet.
3. Open Kaggle, choose **Create**, then **New Notebook**, and then use **File**, then **Import Notebook** to upload `tools/gpu/generated/kaggle_area04.ipynb` (or any other area).
4. Open the **Settings** panel of the notebook. Set **Accelerator** to a GPU and turn **Internet** on.
5. Press **Run All**. A run of one area takes a few minutes.
6. When it finishes, open the **Output** tab and download the file whose name starts with `gpu_checks_cuda`.
7. Check that the file says `"is_real_gpu_run": true`. Then the check really ran on a GPU, and you can write down the result.

## Path B: the Kaggle command line (one command, no clicking)

This path uploads the notebook, runs it on Kaggle's GPU, waits, and downloads the result files for you.

### Step 1: install the Kaggle command line

Run `pip install kaggle` inside the project's virtual environment.
The Kaggle documentation says you need Python 3.11 or newer, and that `pipx install kaggle` also works
([Kaggle CLI documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md)).

### Step 2: get your free Kaggle access token

1. Sign in to Kaggle and open [https://www.kaggle.com/settings/api](https://www.kaggle.com/settings/api).
2. In the **API** section, click the **Generate New Token** button. Kaggle shows you a long secret text.
3. Give that text to the command line in one of these ways. Each way is described in the Kaggle documentation linked above.
   - Way 1 is to run `kaggle auth login` and follow the steps in your web browser.
   - Way 2 is to put the token in an environment variable: `export KAGGLE_API_TOKEN=PASTE_YOUR_TOKEN_HERE`.
   - Way 3 is to save the token in a file called `~/.kaggle/access_token`.
   - Way 4 is the older style: on the same web page, under "Legacy API Credentials", click **Create Legacy API Key**, and save the downloaded `kaggle.json` file as `~/.kaggle/kaggle.json`.
4. Treat the token like a password. Never type it into a file inside this repository, and never upload it to GitHub. The `.gitignore` file of this repository already ignores `kaggle.json` and `access_token` as a safety net.

### Step 3: run one area on a free GPU

```bash
bash tools/gpu/kaggle_run.sh 04 YOUR_KAGGLE_NAME https://github.com/Metamyrddin/ai-eng-toolkit.git
```

What the script does for you:

- It writes a `kernel-metadata.json` file with `"enable_gpu": "true"` and `"enable_internet": "true"`. These field names come from the
  [Kaggle kernel metadata documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md).
- It runs `kaggle kernels push -p <folder>`, which uploads the notebook and starts it ([documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md)).
- It checks `kaggle kernels status <your-name>/<notebook-name>` every 30 seconds until the run is complete.
- It runs `kaggle kernels output <your-name>/<notebook-name> -p <folder>` to download the result files into `tools/gpu/results/kaggle_area04/`.

If you want a different GPU type, the same documentation lists the accelerator names that `kaggle kernels push --accelerator` accepts, such as `NvidiaTeslaT4`.
Some of them are not free or are only offered to people in certain competitions.

## Path C: Google Colab

Google says Colab is free to use and gives access to computing resources including GPUs, but that free GPU access is "heavily restricted",
that limits change over time, and that a free notebook can run for at most 12 hours depending on availability
([Colab FAQ](https://research.google.com/colaboratory/faq.html)).

1. Open [colab.research.google.com](https://colab.research.google.com) and sign in with a Google account.
2. Choose **File**, then **Upload notebook**, and select `tools/gpu/generated/colab_area04.ipynb`. Colab can also open notebooks straight from GitHub: it uses addresses of the form `https://colab.research.google.com/github/YOUR_NAME/ai-eng-toolkit/blob/main/PATH/TO/notebook.ipynb`
   (see Google's [Colab and GitHub demo notebook](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)).
3. In the menu choose **Runtime**, then **Change runtime type**, and pick a GPU. The menu names may change a little over time.
4. Choose **Runtime**, then **Run all**. The last cell downloads the result file to your computer.

## What each area checks on the GPU

| Area | What the GPU check does |
|---|---|
| 01 | It generates text with and without the KV cache and requires the same words both times. It also records how long each way took. |
| 02 | It compares the tiled attention from this toolkit with PyTorch's built-in attention, and it runs the SDPA backend script on the GPU. |
| 03 | With two or more GPUs it trains with DDP over NCCL. With one GPU it reports SKIPPED, and on the CPU it uses the gloo dry run. |
| 04 | It trains LoRA adapters for 15 steps and requires that the loss goes down. |
| 05 | It scores two answers with the language model on the device and checks a statistics helper. |
| 06 | It runs the MinHash duplicate check. This area works on text only, so it always runs on the CPU. |
| 07 | It embeds 300 sentences with the sentence-transformer and checks that related sentences are closer than unrelated ones. |
| 08 | It checks that the logit lens at the last layer equals the model's own output. |

## How to report a GPU result honestly

1. Keep the downloaded JSON file. It contains the GPU name, the PyTorch version and `is_real_gpu_run`.
2. Only call a check "tested on GPU" when `is_real_gpu_run` is `true` and the area status is `PASS`.
3. Copy the file into `tools/gpu/results/` in the repository and mention it in `STATUS.md`.

## Known limits

- The Kaggle command-line steps follow the official documentation, but they have not been run against a real account yet.
- Free GPUs are sometimes busy, so a run may wait in a queue before it starts.
- The vLLM server scripts in `01_inference_serving/cloud/` are not part of these checks, because vLLM replaces PyTorch and is heavy. Run those on a fresh machine as described in that folder.
