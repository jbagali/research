import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM


def get_predictions(input_txt, n_steps, choices_per_step=5):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        #print("Using GPU")
    else:
        device = torch.device("cpu")
        #print("Using CPU")

    #model_name = "gpt2-xl"
    model_name = "shailja/CodeGen_2B_Verilog"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)



    input_ids = tokenizer(input_txt, return_tensors="pt").input_ids.to(device)
    iterations =[]

    with torch.no_grad():
        for i in range(n_steps):
            iteration = dict()
            iteration["Input"] = tokenizer.decode(input_ids[0])
            output = model(input_ids=input_ids)

            # select logits of the first batch and the last token and apply softmax
            next_token_logits = output.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)

            # store tokens with highest probability
            for choice_idx in range(choices_per_step):
                token_id = sorted_ids[choice_idx]
                token_prob = next_token_probs[token_id].cpu().numpy()
                token_choice = (              
                    f"{tokenizer.decode(token_id)} ({100*token_prob:.2f}%)"
                )
                iteration[f"{choice_idx+1}"] = token_choice

            #Append predictions to list
            input_ids = torch.cat([input_ids, sorted_ids[None,0, None]], dim=-1)
            iterations.append(iteration)
            print(iteration)
            #print()

        pd.DataFrame(iterations)


def get_one_prediction(input_txt, n_steps=1, choices_per_step=5):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        #print("Using GPU")
    else:
        device = torch.device("cpu")
        #print("Using CPU")

    #model_name = "gpt2-xl"
    model_name = "shailja/CodeGen_2B_Verilog"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)



    input_ids = tokenizer(input_txt, return_tensors="pt").input_ids.to(device)
    iterations =[]

    with torch.no_grad():
        for i in range(n_steps):
            iteration = dict()
            iteration["Input"] = tokenizer.decode(input_ids[0])
            output = model(input_ids=input_ids)

            # select logits of the first batch and the last token and apply softmax
            next_token_logits = output.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)

            # store tokens with highest probability
            for choice_idx in range(choices_per_step):
                token_id = sorted_ids[choice_idx]
                token_prob = next_token_probs[token_id].cpu().numpy()
                token_choice = (              
                    f"{tokenizer.decode(token_id)} ({100*token_prob:.2f}%)"
                )
                iteration[f"{choice_idx+1}"] = token_choice

            #Append predictions to list
            input_ids = torch.cat([input_ids, sorted_ids[None,0, None]], dim=-1)
            return iteration
            # iterations.append(iteration)
            # print(iteration)
            #print()

        # pd.DataFrame(iterations)

# main 
if __name__ == "__main__":

    #input_txt = "There will be an answer. Let it be"
    #input_txt = "En el Soho hay un cafe donde me escondo"
    input_txt = "def hello_world():"
    n_steps = 1
    choices_per_step = 5

    #get_predictions(input_txt, n_steps, choices_per_step= 10)


    # TODO: here we get the prediction. 
    # modify as needed
    iteration = get_one_prediction(input_txt, n_steps, choices_per_step)
    print(iteration)