import os
import time
import requests
import json
import pdb
from tqdm import tqdm
from bs4 import BeautifulSoup
with open("../data/command_list.txt", "r") as rfile:
    commands = rfile.read().split("\n")[:-1]

RLHF_DIR = "../data/rlhf"
ANSWERS_DIR = "../data/rlhf/answers"

collect_base_json = False
if collect_base_json:
    for cmd in commands:
        r = requests.get("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=activity&answers=3&tagged=linux%3B{}&site=stackoverflow&filter=!nNPvSNOgci".format(cmd))
        with open(os.path.join(RLHF_DIR, f"{cmd}_questions.json"), "w") as wfile:
            json.dump(r.json(), wfile)

def gen_rlhf_dataset(question_title, question_id, answers):
    post_link = f'https://stackoverflow.com/questions/{question_id}/{question_title.lower().replace(" ", "-")}'
    
    r= requests.get(post_link)
    soup = BeautifulSoup(r.text, "html.parser")
    
    dataset = None
    try:
        question_div = soup.find("div", {"class": "postcell"}).find("div", {"class": "s-prose js-post-body"})
        question_children = question_div.find_all(recursive=False)
        question_text = ""
        if question_children[0].name == "div":
            question_children = question_children[1:] # skip div banner
            
        for child in question_children:
            question_text += child.get_text()
        dataset = {"prompt": question_text, "chosen": [], "rejected": []}
    except:
        return None
    
    # no comparison pair
    if len(answers) == 1:
        return None
   
    upvote_average = 0
    answers_with_upvotes = []
   
    votes = soup.find_all("div", {"class": "votecell"})
    answers = soup.find_all("div", {"class": "answercell"})
    if not answers:
        return None
    
    for vote, ans in zip(votes, answers):
        ans_body = ans.find("div", {"class": "s-prose js-post-body"})
        ans_upvotes = vote.find("div", {"class": "js-vote-count"})
            
        answers_with_upvotes.append((ans_body.get_text(), int(ans_upvotes.get_text())))

    num_bad_examples = 1
    
    if len(answers_with_upvotes) <= num_bad_examples:
        return None 
    
    answers_with_upvotes = sorted(answers_with_upvotes, key=lambda x: x[1], reverse=True)
    
    # calculate average
    upvotes_average = sum([ans[1] for ans in answers_with_upvotes]) / len(answers)
    
    # ignore "bad examples" that have better than average upvotes
    while num_bad_examples and answers_with_upvotes[-num_bad_examples][1] > upvote_average:
        num_bad_examples -= 1

    for good_ans in answers_with_upvotes[:-num_bad_examples]:
        for bad_ans in answers_with_upvotes[-num_bad_examples:]:
            dataset["chosen"].append(good_ans[0])
            dataset["rejected"].append(bad_ans[0])
    
    return dataset

def expand_question(dataset):
    dataset["prompt"] = ([dataset["prompt"]] * len(dataset["chosen"]))
    return dataset
        

# gen_rlhf_dataset("using-grep-or-other-tools-for-searching-a-large-csv-file", 77655995, 7254543)
collect_answers = True
if collect_answers:
    for cmd in tqdm(sorted(commands)):
        print(f"scraping {cmd}")
        dataset_path = os.path.join(RLHF_DIR, "answers1", "dataset", f"{cmd}_dataset.json")
        question_path = os.path.join(RLHF_DIR, "answers1", f"{cmd}_questions.json")
        if os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 1000:
            continue
        with open(dataset_path, "w") as wfile:
            datasets = {"prompt": [], "chosen": [], "rejected": []}
            with open(question_path, "r") as rfile:
                question_set = json.load(rfile)
                
                # collect answers to questions according to number of upvotes
                for question in tqdm(question_set["items"]):
                    dataset = gen_rlhf_dataset(question["title"], question["question_id"], question["answers"])
                    # pdb.set_trace()
                    if dataset and dataset["prompt"]:
                        dataset = expand_question(dataset)
                        datasets["prompt"] += dataset["prompt"]
                        datasets["chosen"] += dataset["chosen"]
                        datasets["rejected"] += dataset["rejected"]
                    else:
                        print("nothing returned")
                    time.sleep(3)
            json.dump(datasets, wfile)

