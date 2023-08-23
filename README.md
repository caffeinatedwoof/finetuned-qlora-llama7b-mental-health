# Finetuning of Llama-7B LLM using QLoRA on Mental Health Conversational Dataset

## Features
- Demostration of instruction-tuning on latest open source LLM using a custom dataset on a free colab instance.

## Model Finetuning
A Llama-7B pre-trained model was finetuned using the QLoRA technique on the dataset. The entire finetuning process was done entirely on Google Colab's free-tier using Nvidia T4. 

## Model Inference
PEFT fine-tuned model has been updated here: [caffeinatedwoof/Llama-2-7b-chat-hf-mental-health-conversational_peft](https://huggingface.co/caffeinatedwoof/Llama-2-7b-chat-hf-mental-health-conversational_peft).

## Model History
| Hugging Face Model | Base Model | Dataset | Colab |
| ----------- | ----------- | ----------- | ----------- |
| [llama2-7b-v1](https://huggingface.co/caffeinatedwoof/Llama-2-7b-chat-hf-mental-health-conversational_peft) | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [heliosbrahma/mental_health_conversational_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_conversational_dataset) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/145yWAi1GLuuihYfjEn7mCUYgkcHI4qrb?usp=sharing) |

## References

### Guides / Tutorials / Discussions
- [Fine-tuning of Falcon-7B Large Language Model using QLoRA on Mental Health Conversational Dataset](https://medium.com/@iamarunbrahma/fine-tuning-of-falcon-7b-large-language-model-using-qlora-on-mental-health-dataset-aa290eb6ec85)
- [CRIA, a LLM model series based on Llama 2-7B.](https://github.com/davzoku/cria/blob/main/README.md)
  
### Datasets
- [heliosbrahma/mental_health_chatbot_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset)
- [solomonk/reddit_mental_health_posts](https://huggingface.co/datasets/solomonk/reddit_mental_health_posts)

### Models
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 

