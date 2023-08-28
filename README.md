# Finetuning of Llama-7B LLM using QLoRA on Mental Health Conversational Dataset

## Project Overview ##
Open-source LLMs have impressive capabilities, but they are both large and expensive to train and operate. 

To tackle this issue, I wanted to create a Proof of Concept (POC) using the <a href = https://huggingface.co/meta-llama/Llama-2-7b>Llama-2-7B model</a> that utilises more efficient approaches to finetuning and deployment:

<p align="center">
  <img src="assets/llama-7b.jpeg" alt="Llama-7B Model" width="300">
</p>

### Objectives ###

#### 1. Streamlined Finetuning: ####
Use QLoRA to perform PEFT on Llama-7B model. This significantly reduces memory usage to enable LLM training on a single GPU, offering a more economical alternative.
I demonstrate the instruction-tuning using a custom dataset on a free colab instance.

#### 2. CPU-centric Deployment: ####
Run quantized LLM using open-source libraries such as <a href="https://github.com/ggerganov/llama.cpp">llama.cpp</a> and use only CPU for inference. I also deployed a containerized GGML quantized model to AWS Lambda using AWS CDK and FastAPI.

### Model Finetuning
A Llama-7B pre-trained model was finetuned using the QLoRA technique on the dataset. The entire finetuning process was done entirely on Google Colab's free-tier using Nvidia T4. 

| Hugging Face Model | Base Model | Dataset | Colab | Remarks |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| [llama2-7b-v2](https://huggingface.co/caffeinatedwoof/Llama-2-7b-chat-hf-Amod-mental_health_counseling_conversations) | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/145yWAi1GLuuihYfjEn7mCUYgkcHI4qrb#scrollTo=7xQonNQJSLk0) | Latest |
| [llama2-7b-v1](https://huggingface.co/caffeinatedwoof/Llama-2-7b-chat-hf-mental-health-conversational_peft) | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [heliosbrahma/mental_health_conversational_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_conversational_dataset) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/145yWAi1GLuuihYfjEn7mCUYgkcHI4qrb?usp=sharing) | Experimental |

### Quantizing of Model

To quantize the model, follow these steps:

#### 1. Clone `llama.cpp` Repository and Install Dependencies

Clone the `llama.cpp` repository and install the required dependencies by following these commands:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
python3 -m pip install -r requirements.txt
```

#### 2. Obtain the Original LLaMA Model Weights

Download the original LLaMA model weights and place them in the `./models` directory within our project's structure.

#### 3. Run make_ggml.py Script
Execute the make_ggml.py script with the necessary arguments to perform the quantization. Use the following command:

```bash
python make_ggml.py /path/to/model model-name /path/to/output_directory
```
Replace the placeholders with actual paths and names:

- /path/to/model: Path to your model.
- model-name: Name of the model.
- /path/to/output_directory: Path to the output directory where the quantized model will be saved.

### Deployment
Deploy a container which can run the llama.cpp converted model onto AWS Lambda. We take reference from the <a href="https://github.com/baileytec-labs/llama-on-lambda">OpenLlama on AWS Lambda</a> project, which contains the AWS CDK code to create and deploy a Lambda function leveraging our model of choice, with a FastAPI frontend accessible from a Lambda URL. Note that the free-tier of AWS Lambda gives us 400k GB-s of Lambda Compute each month for free. With proper tuning, this gives us scalable inference of Generative AI LLMs at minimal cost.

We will need the following requirements to get started:
- Docker installed on our system and running.
- AWS CDK installed on our system, as well as an AWS account, proper credentials, etc.
- Python3.9+

Once we've installed the requirements, on Mac/Linux:

```bash
cd ./llama_lambda
chmod +x install.sh
./install.sh
```
Follow the prompts, and when complete, the CDK will provide you with a Lambda Function URL to test the function


## References

### Papers / Repositories / Tutorials / Discussions
- [QLoRA Paper](https://arxiv.org/pdf/2305.14314.pdf)
- [LoRA Paper](https://arxiv.org/pdf/2106.09685.pdf)
- [Fine-tuning of Falcon-7B Large Language Model using QLoRA on Mental Health Conversational Dataset](https://medium.com/@iamarunbrahma/fine-tuning-of-falcon-7b-large-language-model-using-qlora-on-mental-health-dataset-aa290eb6ec85)
- [CRIA, a LLM model series based on Llama 2-7B.](https://github.com/davzoku/cria/blob/main/README.md)
- [Run the LLaMA model using 4-bit integer quantization on a MacBook](https://github.com/ggerganov/llama.cpp)
- [Deploy Serverless Generative AI on AWS Lambda with OpenLLaMa](https://medium.com/@seanbailey518/deploy-serverless-generative-ai-on-aws-lambda-with-openllama-793f97a7cbdd)
  
### Datasets
- [heliosbrahma/mental_health_chatbot_dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset)
- [solomonk/reddit_mental_health_posts](https://huggingface.co/datasets/solomonk/reddit_mental_health_posts)

### Models
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 

