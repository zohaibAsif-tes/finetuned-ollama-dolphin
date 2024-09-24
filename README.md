# Finetuned Ollama Dolphin
Finetuned [Ollama Dolphin 7b](https://huggingface.co/cognitivecomputations/dolphin-2.0-mistral-7b) on custom dataset

# Run Locally
1. Install [`conda`](https://medium.com/@mustafa_kamal/a-step-by-step-guide-to-installing-conda-in-ubuntu-and-creating-an-environment-d4e49a73fc46)
2. Create and activate conda environment using these commands:
```
conda create -n dolphin7b python=3.8.5 anaconda
conda activate dolphin7b
```
3. Install `pip` and the required packages using:
```
sudo apt-get install python3-pip

pip install bitsandbytes --progress-bar off
pip install torch --progress-bar off
pip install -U transformers --progress-bar off
pip install -U peft --progress-bar off
pip install -U accelerate --progress-bar off
pip install loralib --progress-bar off
pip install einops --progress-bar off
pip install scipy
pip install gradio
pip install nvidia-ml-py3
pip install markupsafe==2.0.1
```
4. For training:
```
python3 train.py 
```
5. For testing:
```
python3 test.py
```
