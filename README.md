# Finetuning of Llama-7B LLM using QLoRA on Mental Health Conversational Dataset

## Features
- Demostration of instruction-tuning on latest open source LLM using a custom dataset on a free colab instance.

## Model Finetuning
A sharded Llama-7B pre-trained model was finetuned using the QLoRA technique on the dataset. The entire finetuning process was done entirely on Google Colab's free-tier using Nvidia T4. 

## Model Inference
PEFT fine-tuned model has been updated here: [caffeinatedwoof/Llama-2-7B-bf16-sharded-mental-health-conversational](https://huggingface.co/caffeinatedwoof/Llama-2-7B-bf16-sharded-mental-health-conversational).

## References

### Guides / Tutorials / Discussions
- [Fine-tuning of Falcon-7B Large Language Model using QLoRA on Mental Health Conversational Dataset](https://medium.com/@iamarunbrahma/fine-tuning-of-falcon-7b-large-language-model-using-qlora-on-mental-health-dataset-aa290eb6ec85)
  
### Datasets
- [heliosbrahma/mental_health_chatbot_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset/viewer/default/train?row=0)

### Models
- [TinyPixel/Llama-2-7B-bf16-sharded](https://huggingface.co/TinyPixel/Llama-2-7B-bf16-sharded) 

