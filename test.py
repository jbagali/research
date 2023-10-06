from transformers import AutoTokenizer, CodeGenTokenizerFast, AutoModelForCausalLM
from tree_sitter import Language, Parser
import torch


checkpoint = "Salesforce/codegen-350M-multi"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = CodeGenTokenizerFast.from_pretrained(checkpoint)

#tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-multi')
#model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")

prompt = "def hello_world():"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

#print(tokenizer(prompt)['input_ids'])

#prompt = "#fibonacci.py \n# Implement a function which returns the nth Fibonacci number"

completion = model.generate(
    input_ids,
    num_beams=10, num_return_sequences=10, max_length=10, early_stopping=False,
    return_dict_in_generate=True, output_scores=True)

for i in range(completion["sequences"].shape[0]):
    generated_sentence = tokenizer.decode(completion["sequences"][i], skip_special_tokens=False)
    log_scores = completion['sequences_scores'] 
    normed_prob = (torch.exp(log_scores[i]) / torch.exp(log_scores).sum()).item()
    print(
        f'Ouput {i}: {generated_sentence}\n - normalized probability {normed_prob:.2f}'
    )
 

#print(tokenizer.decode(completion[0], skip_special_tokens=True))
#print(logits)

#@generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True
# total number of tokens 50295
#print({i: tokenizer.decode([i]) for i in range(50)})

# text = "#python program that adds numbers 1 - 10\n def add_1_to_10():"
# #input_ids = tokenizer(text, return_tensors="pt").input_ids
# token_ids = tokenizer.encode(text, return_tensors="pt")
# masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
# masked_pos = [mask.item() for mask in masked_position ]

# with torch.no_grad():
#     output = model(token_ids)
#     #outputs = model.generate(token_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
#     #print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# last_hidden_states = output[0].squeeze()

# list_of_list =[]
# for index,mask_index in enumerate(masked_pos):
#     mask_hidden_state = last_hidden_state[mask_index]
#     idx = torch.topk(mask_hidden_state, k=5, dim=0)[1]
#     words = [tokenizer.decode(i.item()).strip() for i in idx]
#     list_of_list.append(words)
#     print ("Mask ",index+1,"Guesses : ",words)
    

# generated_ids = model.generate(input_ids, max_length=128)
# # print the tokens probability distribution
# print(tokenizer.decode(generated_ids.distribution, skip_special_tokens=True))
#print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

# now tresitter

# add


