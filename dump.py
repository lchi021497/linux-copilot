import os
import pdb
import pymongo
import pandas as pd
from store import store
from explainshell import errors

data_store = store(host='0.0.0.0:27016')
command_list = []
with open("../data/command_list.txt", "r") as f:
  command_list = f.readlines()

def main():
  print("dumping training data from db")
  
  (desc_dataset, rev_desc_dataset) = gen_description_dataset()
  options_dataset = gen_option_dataset()
 
  print("desc dataset len: ", len(desc_dataset))
  print("rev desc dataset len: ", len(rev_desc_dataset))
  print("options dataset len: ", len(options_dataset))

  print("storing dataset to pandas dataframes")
  store_to_file(desc_dataset, "description")
  store_to_file(rev_desc_dataset, "rev_description")
  store_to_file(options_dataset, "options")

def store_to_file(dataset, fname):
  # convert dataset to dataframe
  pd.DataFrame(dataset).to_pickle("{}.pkl".format(fname))

def sub_multi_space_to_one(text):
  it = iter(text)
  filtered_text = ""
  encountered_space = False
  for c in text:
    if c == " ":
      if not encountered_space:
        encountered_space = True
        filtered_text += " "
      continue
    filtered_text += c
    encountered_space = False
  return filtered_text


def gen_description_dataset():
  print("dumping description dataset")
  cmd_query_template = "What does the command {} do?"
  rev_query_template = "What command does {}?"
  cmd_descriptions = []
  rev_cmd_descriptions = []
  for cmd in command_list:
    try:
      result = data_store.findmanpage(cmd.strip())
      if result:
        cmd_descriptions.append({
          "question": cmd_query_template.format(cmd.strip()),
          "answer": result[0].synopsis,
        })
        rev_cmd_descriptions.append({
          "question": rev_query_template.format(result[0].synopsis),
          "answer": cmd.strip(),
        })
        
    except:
      print("error finding manpage for: ", cmd.strip())
  return (cmd_descriptions, rev_cmd_descriptions)

def gen_option_dataset(): 
  print("dumping option dataset")
  cmd_query_template = "What does the option {} of command {} do?"
  response_template = "The options {} of command {} does the following: {}"
  cmd_options = []
  for cmd in command_list:
    try:
      result = data_store.findmanpage(cmd.strip())
      if result:
        for para in result[0].options:
          param_opts = " or ".join(para.opts)

          answer = response_template.format(param_opts, cmd.strip(), para.cleantext().replace("\n", "").replace("\t", ""))
          answer = sub_multi_space_to_one(answer) 
          cmd_options.append({
            "question": cmd_query_template.format(param_opts, cmd.strip()),
            "answer": answer,
          })
    except errors.ProgramDoesNotExist as e:
      print("error finding manpage for: ", cmd.strip())
  return cmd_options

if __name__ == "__main__":
  main()
