from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datetime import datetime
import time, json

class ResponseGenerator():
    def __init__(self, 
                 model_name="Qwen/Qwen2.5-1.5B-Instruct", 
                 context=f"Your name is Rose. You provide one sentence responses. My name is located before the colon or ':'.",
                 max_context=32
                 ):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.max_context = max_context
        self.device = self.model.device
        self.messages = [{"role": "system", "content": f"{context} Today's date is {datetime.now().date()}"}]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.this_dir = str(Path(__file__).parent.resolve())

    def generate_response(self, prompt:str, user_name="Jun", save_to_history=False):
        user_response = {"role": "user", "content": f"{user_name}: {prompt}"}
        self.messages.append(user_response)
        
        if len(self.messages) > self.max_context:
            for _ in range(2): self.messages.pop(1)
        
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generate_start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=192,
            temperature=0.5,
            repetition_penalty = 1.1,
        )
        
        generate_end_time = time.time()
        time_elapsed = generate_end_time - generate_start_time
        print(
                f"[QWEN] \033[93m{time_elapsed:.2f} seconds.\033[0m"
            )
        
        #Returns a tensor() object array result
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        assistant_response = {"role": "assistant", "content": response}
        self.messages.append(assistant_response)
        
        if save_to_history:
            self.__update_json_file([user_response, assistant_response])
        
        print(f"\n\033[092m {response} \033[0m \n")
        return response

    def __update_json_file(self, new_data, filepath=""):
        if not filepath:
            filepath = self.this_dir + '/_qwen_text_history.json'
        
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        # Append new data
        data["messages"].append(new_data)

        # Write the updated data back to the file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)

"""
###DEBUG###
gr = Response_Generator()
while True:
    try:
        prompt = input("> ")
        gr.generate_response(prompt)
    except KeyboardInterrupt:
        break
"""
