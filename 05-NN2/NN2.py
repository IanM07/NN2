import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model_name = "bigscience/bloom-560m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, 
        max_length=50, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0])

print (generate_response("What is the meaning of life?"))
def generate_multiple_responses(prompt, num_return_sequences=2):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids, 
        max_length=50, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_return_sequences=num_return_sequences
        )
    return [tokenizer.decode(output) for output in outputs]

responses = generate_multiple_responses("What is the meaning of life?", num_return_sequences=2)
for i, response in enumerate(responses):
    print(f'Response {i}: {response}')
    
positive_prompts = ["Describe a peaceful day in nature.", "Tell me about the benefits of teamwork.", "What are the joys of learning something new?", "Explain the importance of kindness in society.", "Share an inspiring story of overcoming challenges."]
negative_prompts = ["Describe the consequences of environmental pollution.", "Discuss the impacts of social isolation.", "What are the effects of neglecting health?", "Explain the downsides of excessive competition.", "Narrate a story about losing hope in a difficult situation."] 

#Function to generate and save responses
def generate_and_save_responses(prompts, file_name):
    with open(file_name, "w") as file:
        for prompt in prompts:
            responses = generate_multiple_responses(prompt, num_return_sequences=3)
            for i, response in enumerate(responses):
                file.write(f'Prompt: {prompt}\nResponse {i}: {response}\n\n')

#Generate and save responses
generate_and_save_responses(positive_prompts, "positive_responses.txt")
generate_and_save_responses(negative_prompts, "negative_responses.txt")

