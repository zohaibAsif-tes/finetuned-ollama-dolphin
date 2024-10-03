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
# Files Information

- `train.py:` A python file used for model finetuning through LoRA/QLoRA quantization

- `test.py:` A python file for creating a chatbot with the finetuned model using gradio

- `ogdr-data.txt:` Training data for finetuning the model

- `req.txt:` A file containing all the required libraries 

- `cached_lm_LlamaTokenizer_127_ogdr-data.txt.lock:` A lock file used during the tokenization process to prevent multiple processes from writing to the same cache file simultaneously.

- `cached_lm_LlamaTokenizer_127_ogdr-data.txt:` Cached tokenized data from the reach-chatbot.txt file for faster processing in subsequent runs.

- `finetuned_ogdr_chatbot/adapter_config.json:` Configuration file for the LoRA adapter that contains hyperparameters used during training.

- `finetuned_ogdr_chatbot/adapter_model.safetensors:` The trained model weights for the LoRA adapter in a safetensors format, optimized for efficient storage and loading.

- `ogdr-training-experiments/checkpoint-80/adapter_config.json:` Same as the above config file, but for the specific checkpoint (step 80).

- `ogdr-training-experiments/checkpoint-80/adapter_model.safetensors:` Adapter model weights at checkpoint 80, saved for resuming or analyzing intermediate model states.

- `ogdr-training-experiments/checkpoint-80/optimizer.pt:` State of the optimizer at checkpoint 80, used to resume training from the same point.

- `ogdr-training-experiments/checkpoint-80/rng_state.pth:` State of the random number generator (RNG) at checkpoint 80 to ensure reproducibility.

- `ogdr-training-experiments/checkpoint-80/scheduler.pt:` State of the learning rate scheduler at checkpoint 80 to resume training with the same learning rate progression.

- `ogdr-training-experiments/checkpoint-80/trainer_state.json:` Metadata about the trainer’s state at checkpoint 80, including epoch and step progress.

- `ogdr-training-experiments/checkpoint-80/training_args.bin:` Arguments used for training (like batch size, learning rate) at checkpoint 80, stored for resuming or debugging purposes.


