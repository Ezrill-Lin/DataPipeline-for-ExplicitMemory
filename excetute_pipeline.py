from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from MinHashDeduplication import Deduplicate
from RuleBasedFilter import RuleBasedFilter
from TinyBERTFilter import TinyBERTFilter
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

raw_datasets = {
        "meta-math/MetaMathQA": ['train', 'query', None, False],
        "LDJnr/Capybara": ['train', 'conversation', 'conversation', False],
        "iamtarun/python_code_instructions_18k_alpaca": ['train', 'instruction', None, False],
        "open-r1/OpenR1-Math-220k": ['train', 'problem', None, False],
        "hkust-nlp/CodeIO-PyEdu-Reasoning": ['train', 'prompt', None, True],
    }

def run_filter(raw_datasets):
    for name, l in tqdm(raw_datasets.items()):
        streaming = l[3]
        if not streaming:
            ds = load_dataset(name, split=l[0])
        else:
            ds = []
            ds_stream = load_dataset(name, split=l[0], streaming=True)
            for i, sample in enumerate(ds_stream):
                if i >= 100000:
                    break
                ds.append(sample)
        
        dd_filter = Deduplicate(ds, l[1])
        ds_dd = dd_filter.run()
        rb_filter = RuleBasedFilter(ds_dd, l[2])
        ds_rb = rb_filter.run()
        ds_cleaned = Dataset.from_list(ds_rb)
        ds_cleaned.save_to_disk(f'./datasets/{name}_cleaned')
        print(f'Dataset {name} cleaning completed. \n')

def dataset_concat(raw_datasets, samp=False):
    '''
    Used to combine different datasets into one integrated dataset

    samp (bool): Default is False. 
                If True, only the first 2000 pieces of data entries will be combined.
    '''
    ds_list = []
    for name in tqdm(raw_datasets.keys(), desc='Concatnating'):
        if not samp:
            ds_list.append(load_from_disk(f'./datasets/{name}_cleaned'))
        else:
            ds_list.append(load_from_disk(f'./datasets/{name}_cleaned').select(range(2000)))

    ds_concat = concatenate_datasets(ds_list)
    ds_concat = ds_concat.add_column('index', range(len(ds_concat)))

    return ds_concat

def add_text_or_dataset(sample, add_text=True):
    '''
    Add "text" key and "dataset" key used for TinyBERT classification and 
    keeping a track of where each data is from
    '''
    if sample['query']:
        text = sample['query'] + sample['response']
        dataset = 'metamath'
    elif sample['conversation']:
        text = []
        for t in sample['conversation']:
            text.append(" ".join(t.values()))
        text = " ".join(text)
        dataset = 'capybara'
    elif sample['instruction']:
        text = sample['prompt']
        dataset = 'code18k'
    elif sample['problem']:
        text = sample['problem'] + sample['solution'] + sample['answer']
        dataset = 'openmath'
    elif sample['turn_1']:
        try:
            text = sample['prompt'] + sample['turn_1'] + sample['feedback_1'] + sample['turn_2'] + sample['feedback_2']
        except:
            text = sample['prompt'] + sample['turn_1'] + sample['feedback_1']
        dataset = 'codeio'
    else:
        text = None
        dataset = None
    if add_text:
        return {'text': f'{text}',
                'dataset': f'{dataset}'}
    else:
        return {'dataset': f'{dataset}'}

def sharegpt_format(ds_filtered):
    '''
    Convert the filtered dataset to ShareGPT format
    This function is specific to the dataset used in ExplicitMemory project.
    It can not be used on other datasets
    '''
    ds_formated = []
    for sample in tqdm(ds_filtered, desc='Formatting'):
        if sample['query']:
            ds_formated.append({
                'data_type': sample['dataset'],
                'conversations':[{
                    'from': 'human',
                    'value': sample['query']
                }, {
                    'from': 'gpt',
                    'value': sample['response']
                }]
            })
        elif sample['conversation']:
            conversation = []
            for dialogue in sample['conversation']:
                conversation.append({
                    'from': 'human',
                    'value': dialogue['input']
                })
                conversation.append({
                    'from': 'gpt',
                    'value': dialogue['output']
                })
            ds_formated.append({
                'data_type': sample['dataset'],
                'conversations': conversation
            })
        elif sample['instruction']:
            prompt = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
            ds_formated.append({
                'data_type': sample['dataset'],
                'conversations': [{
                    'from': 'human',
                    'value': prompt + "###Instruction:\n" + sample['instruction'] + "###Input:\n" + sample['input']
                }, {
                    'from': 'gpt',
                    'value': sample['output']
                }]
            })
        elif sample['problem']:
            ds_formated.append({
                'data_type': sample['dataset'],
                'conversations': [{
                    'from': 'human',
                    'value': sample['problem']
                }, {
                    'from': 'gpt',
                    'value': sample['solution']
                }]
            })
        elif sample['turn_1']:
            if sample['turn_2']:
                ds_formated.append({
                    'data_type': sample['dataset'],
                    'conversations': [{
                        'from': 'human',
                        'value': sample['prompt']
                    }, {
                        'from': 'gpt',
                        'value': sample['turn_1']
                    }, {
                        'from': 'human',
                        'value': sample['feedback_1']
                    }, {
                        'from': 'gpt',
                        'value': sample['turn_2']
                    }]
                })
            else:
                ds_formated.append({
                    'data_type': sample['dataset'],
                    'conversations': [{
                        'from': 'human',
                        'value': sample['prompt']
                    }, {
                        'from': 'gpt',
                        'value': sample['turn_1']
                    }]
                })

    with open('sftdata.json', 'w', encoding='utf-8') as f:
        json.dump(ds_formated, f, indent=4)

def start_execution(raw_datasets):
    '''
    Go through all pipelines to produce reasoning enhanced SFT data.
    '''
    _ = run_filter(raw_datasets)

    samp_to_label = dataset_concat(raw_datasets, samp=True)
    samp_to_label.save_to_disk('./datasets/sample_to_label')

    all_data = dataset_concat(raw_datasets)
    all_data = all_data.map(lambda sample: add_text_or_dataset(sample=sample, add_text=False))
    all_data.save_to_disk('./datasets/all_data')

    all_data_shuffled6 = all_data.shuffle(seed=6)
    texts = all_data_shuffled6.map(add_text_or_dataset)
    texts_extracted = texts.remove_columns([col for col in texts.column_names \
                                            if col != 'text' and col != 'dataset' and col != 'index'])

    with open('./jsonl/all_data_to_bert.jsonl', 'w', encoding='utf-8') as f:
        for text in tqdm(texts_extracted):
            f.write(json.dumps(text) + "\n")
    
    tb_filter = TinyBERTFilter(model_path='./tinybert-filter',
                           data_to_label='./jsonl/all_data_to_bert.jsonl')
    _ = tb_filter.start_label()
    _ = tb_filter.save_labelled_data()

    ds_origin = load_from_disk('./datasets/all_data')
    filtered_data = load_from_disk('./datasets/labelled_data_filtered')
    indices = filtered_data['index']
    ds_filtered = ds_origin.select(indices)

    _ = sharegpt_format(ds_filtered)

