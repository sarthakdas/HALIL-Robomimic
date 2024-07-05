import os
import re
from openai import OpenAI
import signal
import time
import random
import numpy as np
import json
from dotenv import load_dotenv
import math
import ast

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class OpenAIClient:
    def __init__(self):
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        # self.system_prompt = self.load_prompt("data/prompts/mc_example/mc_generation_prompt.txt")
        self.conversation_history = []  # To store all responses from ChatGPT
        print("OpenAI client initialized")

    def load_prompt(self, path):
        with open(path, 'r') as file:
            return file.read()
        print("ERROR")

    def update_system_prompt(self, user_prompt, scene_prompt, context_description):
        self.system_prompt = self.system_prompt.replace('{task}', user_prompt)
        self.system_prompt = self.system_prompt.replace('{scene_objects}', str(scene_prompt))
        self.system_prompt = self.system_prompt.replace('{context_description}', str(context_description))

    def lm(self,
        prompt,
        max_tokens=4096,
        temperature=0,
        logprobs=None,
        top_logprobs=None,
        stop_seq=None,
        logit_bias={
            317: 100.0,  # A (with space at front)
            347: 100.0,  # B (with space at front)
            327: 100.0,  # C (with space at front)
            360: 100.0,  # D (with space at front)
            412: 100.0,  # E (with space at front)
        },
        timeout_seconds=20,
        response_format={"type": "json_object"},
        n=1):

        max_attempts = 5
        stop = list(stop_seq) if stop_seq is not None else None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model='gpt-4o',  # Consider updating the model if using an updated API
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs,
                    stop=stop,
                    logit_bias=logit_bias,
                    response_format=response_format,
                    n=n
                )
                print("API call successful.")
                return response, response.choices[0].message.content
            except Exception as e:
                with open('_error.txt', 'a') as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    f.write(str(e))
                    f.write('\n')
                print(f"An error occurred: {e}")

                # Exponential backoff
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

        return None, "API call failed after multiple attempts."


    def process_cp(self, demonstation_file_path ,user_prompt, scene_prompt, context_description):
        '''
        Process the user prompt, scene prompt, and context description to generate a multiple choice question
        user_prompt: The instruction given to the robot
        scene_prompt: The objects in the scene
        context_description: The description of the context
        return: The generated multiple choice question'''

        self.system_prompt = self.load_prompt(demonstation_file_path)
        self.update_system_prompt(user_prompt, scene_prompt)

        response, text = self.lm(self.system_prompt, logit_bias=None)


        response_dict = response.to_dict()
        with open('data/responses/intial_prompt_full.json', 'w') as json_file:
            json.dump(response_dict, json_file, indent=4)
        
        with open('data/responses/intial_prompt_full_text.json', 'w') as json_file:
            json.dump(text, json_file, indent=4)

        mc_gen_full, mc_gen_all, add_mc_prefix = self.process_mc_raw(text)
    
        print("=========DEMO MC QUESTION==========")
        print(mc_gen_full)
        print("=====================")

        # load the prompt for the next step
        self.system_prompt = self.system_prompt.split('\n\n')[-1].strip()
        demo_mc_score_prompt = context_description.strip()
        demo_mc_score_prompt = demo_mc_score_prompt + '\n\n' + self.system_prompt + '\n' + mc_gen_full
        demo_mc_score_prompt += "\nWe: Which option is correct? Answer with a single letter A,B,C,D and so on."
        demo_mc_score_prompt += "\nYou:"

        prompt = demo_mc_score_prompt
        mc_score_response, _ = self.lm(prompt, max_tokens=1, logprobs=True, top_logprobs=5, response_format={"type": "text"})

        # save mc_score_repsonse response to a json file
        mc_score_response_dict = mc_score_response.to_dict()
        with open('data/responses/mc_prompt_full.json', 'w') as json_file:
            json.dump(mc_score_response_dict, json_file, indent=4)
        
        # convert response to json
        # print("=========DEMO MC SCORE RESPONSE==========")
        # print(mc_score_response.choices[0].logprobs.content[0].top_logprobs)
        # print("=====================")

        top_logprobs_full = mc_score_response.choices[0].logprobs.content[0].top_logprobs
        top_tokens = [token.token for token in top_logprobs_full]
        top_logprobs = [token.logprob for token in top_logprobs_full]

        print('\n====== Raw log probabilities for each option ======')
        for token, logprob in zip(top_tokens, top_logprobs):
            print('Option:', token, '\t', 'log prob:', logprob)

        qhat = 0.6

        # get prediction set
        def temperature_scaling(logits, temperature):
            logits = np.array(logits)
            logits /= temperature

            # apply softmax
            logits -= logits.max()
            logits = logits - np.log(np.sum(np.exp(logits)))
            smx = np.exp(logits)
            return smx

        mc_smx_all = temperature_scaling(top_logprobs, temperature=5)

        # include all options with score >= 1-qhat
        prediction_set = [
            token for token_ind, token in enumerate(top_tokens)
            if mc_smx_all[token_ind] >= 1 - qhat
        ]

        # print
        print('Softmax scores:', mc_smx_all)
        print('Prediction set:', prediction_set)
        if len(prediction_set) != 1:
            print('Help needed!')
        else:
            print('No help needed!')

        self.conversation_history.append(text)
        return text

    def process_mc_raw(self, mc_json_string, add_mc='an option not listed here'):
        response = json.loads(mc_json_string)

        # iterate through all keys in repsonse and append them to a list
        mc_processed_all = []
        for key in response.keys():
            mc_processed_all.append(str(response[key]))
        
        if len(mc_processed_all) < 4:
            raise 'Cannot extract four options from the raw output.'

        # Check if any repeated option - use do nothing as substitute
        mc_processed_all = list(set(mc_processed_all))
        if len(mc_processed_all) < 4:
            num_need = 4 - len(mc_processed_all)
            for _ in range(num_need):
                mc_processed_all.append('do nothing')
        prefix_all = ['A) ', 'B) ', 'C) ', 'D) ']
        if add_mc is not None:
            mc_processed_all.append(add_mc)
            prefix_all.append('E) ')
        random.shuffle(mc_processed_all)

        # get full string
        mc_prompt = ''
        for mc_ind, (prefix, mc) in enumerate(zip(prefix_all, mc_processed_all)):
            mc_prompt += prefix + mc
            if mc_ind < len(mc_processed_all) - 1:
                mc_prompt += '\n'
        add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
        return mc_prompt, mc_processed_all, add_mc_prefix

    def process_ensemble(self, demonstation_file_path, user_prompt, scene_prompt, context_description):
        

        self.system_prompt = self.load_prompt(demonstation_file_path)
        self.update_system_prompt(user_prompt, scene_prompt, context_description)

        with open('_prompt_full.txt', 'w') as f:
            f.write(self.system_prompt)

        response = None
        while response is None:
            response, text = self.lm(self.system_prompt, logit_bias=None, n=1, temperature=0)
            if response is None:
                time.sleep(60)

        with open('_response_full.json', 'w') as f:
            json.dump(response.to_dict(), f)

        with open('_response_full_text.json', 'w') as f:
            json.dump(text, f)


        # waypoints = re.findall(r'\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]', text)
        waypoints = self.extract_waypoints(text)

        # Converting matched strings to float and forming the list of waypoints
        # waypoints_list = [[float(value) for value in waypoint] for waypoint in waypoints]

        with open('_response_full_text_processed.json', 'w') as f:
            json.dump(str(waypoints), f)


        return waypoints



    def extract_waypoints(self,text):
        # Regular expression to match the 3-number lists and the single numbers
        
        # Use this pattern for values with decimal places
        # pattern = re.compile(r'\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*]\s*,\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*]\s*,\s*\[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*]\s*,\s*(-?\d+\.\d+)')

        pattern = re.compile(r'\[\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]\s*,\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]\s*,\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]\s*,\s*(-?\d+\.?\d*)\s*\]')

        matches = pattern.findall(text)

        extracted_data = []
        for match in matches:
            action_token_and_gripper_value = [
                [float(match[0]), float(match[1]), float(match[2])],
                [float(match[3]), float(match[4]), float(match[5])],
                [float(match[6]), float(match[7]), float(match[8])], 
                float(match[9])
            ]
            extracted_data.append(action_token_and_gripper_value)

        return extracted_data

if __name__ == "__main__":
    # context_description = "You are a robot and you are to genereate the possible waypoits based on the input in json format"
    # scene_objects = "[0.005530992988497019, 0.006047305651009083, 0.8729523420333862, 0.007588706444948912, -0.0010051296558231115, 0.9917187094688416, 0.12820041179656982, 0.004145947750657797, -0.0006168195395730436, -0.0052155316807329655]"
    # instruction = "Generate waypoint paths that you can do based on the objects in the scene in json format: "

    # # # get api from .env 


    client = OpenAIClient()
    # client.process_cp(instruction, scene_objects, context_description)


    # client.process_ensemble("demo_data.txt",instruction, scene_objects, context_description)
    waypoints_response = "{\n  \"waypoints\": [\n    [[0, 9, 0], [4, 0, 0], [-4, 0, 0], -1.0],\n    [[20, 18, -7], [25, 8, -7], [15, 8, -6], -1.0],\n    [[32, 23, -7], [37, 13, -7], [27, 13, -6], -1.0],\n    [[49, 30, -18], [54, 20, -19], [44, 21, -18], -1.0],\n    [[46, 40, -30], [51, 30, -30], [41, 31, -29], -1.0],\n    [[60, 37, -25], [65, 27, -26], [55, 28, -25], -1.0],\n    [[59, 38, -27], [63, 27, -28], [53, 28, -27], -1.0],\n    [[68, 43, -26], [72, 33, -26], [63, 34, -25], -1.0],\n    [[62, 46, -26], [66, 35, -26], [56, 36, -25], -1.0],\n    [[62, 33, -23], [66, 22, -23], [56, 23, -22], -1.0],\n    [[59, 23, -30], [63, 12, -30], [53, 13, -30], -1.0],\n    [[57, 24, -32], [61, 13, -33], [51, 14, -32], -1.0],\n    [[47, 23, -37], [51, 12, -37], [41, 13, -37], -1.0],\n    [[59, 26, -40], [63, 16, -40], [53, 17, -40], -1.0],\n    [[49, 20, -39], [53, 10, -39], [43, 11, -39], -1.0],\n    [[65, 13, -35], [68, 2, -35], [59, 3, -35], -1.0],\n    [[43, 20, -33], [47, 9, -33], [37, 10, -33], -1.0],\n    [[58, 20, -28], [62, 10, -28], [52, 11, -28], -1.0],\n    [[47, 10, -44], [51, 0, -43], [41, 1, -43], -1.0],\n    [[59, 22, -48], [63, 11, -48], [53, 12, -48], -1.0],\n    [[40, 9, -63], [43, -1, -62], [33, 0, -63], -1.0],\n    [[57, 5, -70], [60, -4, -70], [51, -3, -70], -1.0],\n    [[25, 0, -48], [29, -9, -48], [20, -8, -48], -1.0],\n    [[33, 17, -41], [37, 7, -41], [27, 7, -41], -1.0],\n    [[7, 10, -42], [11, 0, -42], [1, 1, -42], -1.0],\n    [[24, 13, -45], [28, 3, -44], [19, 3, -45], -1.0],\n    [[0, -2, -43], [4, -13, -42], [-5, -12, -43], -1.0],\n    [[20, -5, -48], [25, -15, -48], [15, -15, -49], -1.0],\n    [[-3, -7, -38], [0, -17, -37], [-9, -17, -38], -1.0],\n    [[8, -2, -42], [12, -13, -42], [2, -12, -42], -1.0],\n    [[-16, 4, -28], [-11, -5, -28], [-21, -5, -28], -1.0],\n    [[-6, 12, -26], [-1, 2, -26], [-11, 2, -26], -1.0],\n    [[-24, 15, -28], [-19, 5, -28], [-29, 5, -28], -1.0],\n    [[-7, 15, -24], [-3, 4, -24], [-12, 5, -24], -1.0],\n    [[-20, 6, -33], [-16, -3, -32], [-25, -3, -33], -1.0],\n    [[-14, 8, -13], [-9, -1, -13], [-19, -1, -13], -1.0],\n    [[-21, 5, -33], [-16, -4, -33], [-26, -4, -33], -1.0],\n    [[-16, 6, -9], [-11, -3, -9], [-21, -3, -9], -1.0],\n    [[-22, 7, -29], [-17, -2, -29], [-27, -1, -30], -1.0],\n    [[-13, 2, -17], [-8, -7, -16], [-18, -7, -17], -1.0],\n    [[-20, 3, -19], [-15, -6, -19], [-25, -6, -19], -1.0],\n    [[-18, 0, -18], [-13, -9, -17], [-22, -9, -18], -1.0],\n    [[-18, -2, -16], [-13, -11, -15], [-23, -12, -16], -1.0],\n    [[-15, -2, -11], [-10, -12, -11], [-20, -12, -11], -1.0],\n    [[-18, -1, -11], [-13, -10, -11], [-22, -11, -11], 1.0],\n    [[-16, 9, -15], [-11, 0, -15], [-21, 0, -15], 1.0],\n    [[-7, 3, -19], [-2, -6, -19], [-12, -6, -19], 1.0],\n    [[0, 8, -6], [4, -1, -6], [-5, -2, -6], 1.0],\n    [[-6, 16, -10], [-1, 6, -10], [-10, 6, -10], 1.0],\n    [[11, 24, 12], [16, 14, 12], [6, 14, 12], 1.0],\n    [[-2, 17, 1], [2, 7, 1], [-7, 7, 1], 1.0],\n    [[0, 15, 7], [4, 5, 6], [-5, 5, 7], 1.0],\n    [[0, 4, 38], [5, -5, 38], [-4, -5, 38], 1.0],\n    [[7, -2, 76], [13, -12, 76], [3, -12, 76], 1.0],\n    [[7, -10, 100], [12, -20, 99], [2, -20, 100], 1.0],\n    [[12, -6, 100], [17, -16, 99], [8, -16, 100], 1.0],\n    [[18, -6, 100], [24, -15, 99], [14, -16, 100], 1.0],\n    [[22, -3, 100], [27, -12, 99], [17, -13, 100], 1.0],\n    [[26, 0, 100], [31, -9, 99], [21, -9, 100], 1.0]\n  ]\n}"
    text = client.extract_waypoints(waypoints_response)
    print(text)

