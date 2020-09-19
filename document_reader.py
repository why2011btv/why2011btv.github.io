from transformers import RobertaTokenizer, RobertaModel
import os
import torch
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import sent_tokenize
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')
import spacy
nlp = spacy.load("en_core_web_sm")
#model = RobertaModel.from_pretrained('roberta-base')
space = ' '
#dir_name = "/shared/why16gzl/logic_driven/Quizlet/Quizlet_2/LDC2020E20_KAIROS_Quizlet_2_TA2_Source_Data_V1.0/data/ltf/ltf/"
#file_name = "K0C03N4LR.ltf.xml"

def RoBERTa_list(content, token_list = None, token_span_SENT = None):
    encoded = tokenizer.encode(content)
    roberta_subword_to_ID = encoded
    # input_ids = torch.tensor(encoded).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    roberta_subwords = []
    roberta_subwords_no_space = []
    for index, i in enumerate(encoded):
        r_token = tokenizer.decode([i])
        roberta_subwords.append(r_token)
        if r_token[0] == " ":
            roberta_subwords_no_space.append(r_token[1:])
        else:
            roberta_subwords_no_space.append(r_token)

    roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1:-1]) # w/o <s> and </s>
    roberta_subword_map = []
    if token_span_SENT is not None:
        roberta_subword_map.append(-1) # "<s>"
        for subword in roberta_subword_span:
            roberta_subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
        roberta_subword_map.append(-1) # "</s>" 
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map
    else:
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, -1

"""
# Problem with previous version of RoBERTa_list (roberta_list): "shoal" will be splitted into "sho" and "al"; the word after "shoal" is "and"; so an assertion error will be thrown
def roberta_list(content, token_list):
    encoded = tokenizer.encode(content)
    roberta_subword_to_ID = encoded
    # input_ids = torch.tensor(encoded).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    roberta_subwords = []
    roberta_subword_map = []
    token_count = -1
    for index, i in enumerate(encoded):
        token = tokenizer.decode([i])
        roberta_subwords.append(token)
        if token[0] == " ": # subword with space as first char always denotes the start of a new token
            token_count += 1
            assert token_list[token_count][0] == token[1]
        else:
            # Case 1 token after quotation mark: "My ... >>> [' "', 'My', ...]
            # Case 2 punctuation: He cried, and ... >>> [' He', ' cried', ',', ' and', ...]
            if (token_count + 1) < len(token_list) and token[0] == token_list[token_count + 1][0]:
                token_count += 1
            else:
                # token_count remains the same unless the subword is end of sentence
                if token == "</s>":
                    token_count = -1
                roberta_subword_map.append(token_count)
    return roberta_subword_to_ID, roberta_subwords, roberta_subword_map
"""

def ltf_reader(dir_name, file_name, token_provided = True):
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".ltf.xml", "") # e.g., K0C03N4LR
    my_dict["sentences"] = []
    my_dict["doc_content"] = ""
    tree = ET.parse(dir_name + file_name)
    root = tree.getroot()
    for child in root:
        for TEXT in child:
            for SEG in TEXT:
                sent_dict = {} # one dict for each sentence
                sent_dict["sent_id"] = int(SEG.attrib['id'].replace("segment-", '')) # e.g., segment-0
                sent_dict["content"] = SEG[0].text # content of sentence
                sent_dict["sent_start_char"] = seg_start = \
                int(SEG.attrib["start_char"]) # position of start char of sentence in the doc
                sent_dict["sent_end_char"] = int(SEG.attrib["end_char"])
                
                # Recover complete original text
                if len(my_dict["doc_content"]) <= seg_start:
                    my_dict["doc_content"] += space * (seg_start - len(my_dict["doc_content"]))
                else:
                    raise ValueError("Impossible situation arises.")
                my_dict["doc_content"] += sent_dict["content"] 
                
                if token_provided == True:
                    sent_dict["token_span_DOC"] = [] # token spans in the document level, e.g., (116, 126)
                    sent_dict["token_span_SENT"] = [] # token spans in the sentence level, e.g., (2, 12)
                    sent_dict["tokens"] = [] # tokens, e.g., 20-year-old
                    # Read ltf-tokenized tokens
                    temp_count = 0 # SEG[0] is content of sentence; SEG[1], SEG[2], ... are tokens
                    for TOKEN in SEG:
                        if temp_count > 0: # Not including SEG[0]
                            sent_dict["token_span_DOC"].append([int(TOKEN.attrib["start_char"]), int(TOKEN.attrib["end_char"])])
                            sent_dict["token_span_SENT"].append([int(TOKEN.attrib["start_char"]) - seg_start, int(TOKEN.attrib["end_char"]) - seg_start])
                            sent_dict["tokens"].append(TOKEN.text)
                        temp_count += 1

                    # Part Of Speech Tagging
                    sent_dict["pos"] = nltk.pos_tag(sent_dict["tokens"])

                    # RoBERTa tokenizer
                    sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
                    sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
                    RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
                    
                    sent_dict["roberta_subword_span_DOC"] = \
                    span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
                else:
                    sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
                    sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
                    RoBERTa_list(sent_dict["content"])
                    
                    sent_dict["roberta_subword_span_DOC"] = \
                    span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
                my_dict["sentences"].append(sent_dict)
    return my_dict

def tokenized_to_origin_span(text, token_list):
    token_span = []
    pointer = 0
    for token in token_list:
        while True:
            if token[0] == text[pointer]:
                start = pointer
                end = start + len(token) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span.append([start, end])
    return token_span

def sent_id_lookup(my_dict, start_char, end_char):
    for sent_dict in my_dict['sentences']:
        if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
            return sent_dict['sent_id']

def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index
        
label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
num_dict = {0: "SuperSub", 1: "SubSuper", 2: "Coref", 3: "NoRel"}
def label_to_num(label):
    return label_dict[label]
def num_to_label(num):
    return num_dict[num]

def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC

def roberta_subword_id_lookup(roberta_subword_span_SENT, start_char):
    # subword_id: start from 0
    subword_id = -1
    for subword_span in roberta_subword_span_SENT:
        subword_id += 1
        if subword_span[0] == start_char:
            return subword_id
    raise ValueError("No subword is found.")
    return subword_id
    
def tsvx_reader(dir_name, file_name, token_provided = False):
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".tsvx", "") # e.g., article-10901.tsvx
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    
    # Read tsvx file
    for line in open(dir_name + file_name):
        line = line.split('\t')
        if line[0] == 'Text':
            my_dict["doc_content"] = line[1]
        elif line[0] == 'Event':
            end_char = int(line[4]) + len(line[2]) - 1
            my_dict["event_dict"][int(line[1])] = {"mention": line[2], "start_char": int(line[4]), "end_char": end_char} 
            # keys to be added later: sent_id & subword_id
        elif line[0] == 'Relation':
            event_id1 = int(line[1])
            event_id2 = int(line[2])
            rel = label_to_num(line[3])
            my_dict["relation_dict"][(event_id1, event_id2)] = {}
            my_dict["relation_dict"][(event_id1, event_id2)]["relation"] = rel
        else:
            raise ValueError("Reading a file not in HiEve tsvx format...")
    
    # Split document into sentences
    sent_tokenized_text = sent_tokenize(my_dict["doc_content"])
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        if token_provided == True:
            # NLTK-tokenized tokens & Part-Of-Speech Tagging
            for token in spacy_token:
                sent_dict["tokens"].append(token.text)
                sent_dict["pos"].append(token.pos_)
            sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
            sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

            # RoBERTa tokenizer
            sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
            sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
            RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
            sent_dict["roberta_subword_span_DOC"] = \
            span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        else:
            sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
            sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
            RoBERTa_list(sent_dict["content"])
            
            sent_dict["roberta_subword_span_DOC"] = \
            span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        my_dict["sentences"].append(sent_dict)
    
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
        sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = \
        roberta_subword_id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"])
    return my_dict