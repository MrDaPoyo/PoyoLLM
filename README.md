# PoyoSLM!
PoyoSLM is a simple, small (124M params as of right now) language model designed for conversational tasks. It was trained from scratch on a single Nvidia L4 GPU using TikToken and PyTorch.

## How do I train mine?
To train your own PoyoSLM, you can use the `train_model.py` script. It will create a new weights file in the `log` directory. It will automatically save its progress. But before doing that, you need to prepare your data. For that purpose, please run `fineweb.py` before anything else. BEWARE!!! It will take a few gigabytes of disk space, so make sure you have enough free space <3.

## How do I finetune it?
To finetune PoyoSLM, you can use the `finetuning.py` script. It will try to train the model on a few conversations from a dataset from Huggingface. It will also save the weights in the `log` directory. After that, you'll be ready to use it for whatever you want!

## Example inputs (from the web UI):
- "What is the meaning of life?"
> I can't give an answer at all; I'm not a language model.

- "I hate my job"
> Yes, I am fine. I'm very smart of getting ready. I am very smart of that. My great.

- "I want to learn a new language"
> Could you please try out multiple weeks of prompts and follow the same rules as it is possible to get the same amount of prompts and an outline of your request? I will provide some recommendations as to where I can assist you!

- "Moron"
> I am a language model. I have tried to write my previous sentence and not I have not added my previous sentence. How do I know if I may say "the name of the character of the character of a person who is a "competatory force"? Does all of this mean I am a language model? I am not just one language model. They are not part of the original word or word order. They are part of the original word order which the person will call. In this way I can't be sure if I know no the names of those.

As you may have told, these outputs make no sense whatsoever. This is because the model is soooo tiny (awww :3) that it can't really learn anything useful. But it is still fun to play with!