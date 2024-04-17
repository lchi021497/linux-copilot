import os
import subprocess
from infer import infer_model, infer_pipe
from common import format_test_example, extract_result

ASK_MODE = 0
EXEC_MODE = 1

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
            # prompt LLM for command info
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
            
