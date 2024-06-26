import os
import subprocess
import requests
import json
from common import format_test_example, extract_result

ASK_MODE = 0
EXEC_MODE = 1

USE_LLAMA_BACKEND=False

# fall back to use huggingface inference pipeline
if not USE_LLAMA_BACKEND:
    from infer import infer_model, infer_pipe

def main():
    mode = EXEC_MODE 
    while True:
        if mode == EXEC_MODE:
            input_text = input("exec$ ")
        else:
            input_text = input("ask$ ")
            
        if input_text == "ask":
            if mode == ASK_MODE:
                print("already in ask mode")
            mode = ASK_MODE
            continue
        elif input_text == "exec":
            if mode == EXEC_MODE:
                print("already in exec mode")
            mode = EXEC_MODE
            continue
        
        if mode == ASK_MODE:
            # call llama-server backend
            if USE_LLAMA_BACKEND:
                # prompt LLM for command info
                input_prompt = {'prompt': format_test_example(input_text)}
                resp = requests.post('http://localhost:8080/completion', headers={'Content-Type':'application/json'}, data=json.dumps(input_prompt))
                print('> ', resp.json()['content'])
            else:
                # use hugginface inference pipeline
                result = infer_pipe(format_test_example(input_text))
                print(extract_result(result[0]["generated_text"]))
        else:
            # execute command in a subshell
            p = subprocess.Popen(input_text, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True)
            #for shell=False use absolute paths
            p_stdout = p.stdout.read()
            p_stderr = p.stderr.read()
            print(p_stdout)
if __name__ == "__main__":
    main()
            
