import re
import numpy as np
from datasets import load_dataset
from collections import Counter
from scipy.stats import entropy
from typing import Optional
from typing import List, Dict, Any 

class RuleBasedFilter():
    def __init__(self, dataset: List[Dict[str, Any]], 
                 based_on: Optional[str]=None):
        '''
        Difference from MinHash filtering which only applies to question,
        rule-based filtering applies the whole document

        Except for "Capybara" dataset who need to specify based_on="conversation",
        all other dataset do not need to specify based_on argument
        '''
        self.dataset = dataset
        self.based_on = based_on
    
    def word_count(self, text: str) -> int:
        """Returns the number of words in a text."""
        words = re.findall(r'\w+', text)
        return len(words)

    def mean_word_length(self, text: str) -> int:
        """Returns the mean word length in a text."""
        words = re.findall(r'\w+', text)
        if len(words) == 0:
            return 0
        return np.mean([len(word) for word in words])

    def non_alpha_fraction(self, text: str) -> int:
        """Returns the fraction of non-alphabetic characters in the text."""
        non_alpha_chars = sum(1 for char in text if not char.isalpha() and not char.isspace())
        total_chars = len(text)
        return non_alpha_chars / total_chars if total_chars > 0 else 0

    def unique_word_fraction(self, text: str) -> int:
        """Returns the fraction of unique words in the text."""
        words = re.findall(r'\w+', text.lower()) 
        if len(words) == 0:
            return 0
        unique_words = set(words)
        return len(unique_words) / len(words)

    def compute_unigram_entropy(self, text: str) -> int:
        """Computes entropy based on unigram word distribution in the text."""
        words = re.findall(r'\w+', text.lower()) 
        word_counts = Counter(words)
        total_words = sum(word_counts.values())
        word_probs = [count / total_words for count in word_counts.values()]
        return entropy(word_probs, base=2) 

    def run(self) -> List[Dict[str, Any]]:
        """Applies rule-based filtering and returns a filtered dataset."""
        filtered_data = []
        
        for sample in self.dataset:
            if not self.based_on:
                try: text = " ".join(sample.values())  # filter applied to the whole document 
                except: # in case where sample.values() has non-string objects
                    samp = [s for s in sample.values() if isinstance(s, str)]
                    text = " ".join(samp)
            else:
                text = " ".join(sample[self.based_on][0].values())

            if self.word_count(text) < 50:
                continue  # Skip short documents
            if self.mean_word_length(text) > 10:
                continue  # Skip long-word documents
            if self.non_alpha_fraction(text) >= 0.7:
                continue  # Skip documents with too many non-alphabetic characters
            if self.unique_word_fraction(text) > 0.8:  
                continue  # Skip low-quality documents with excessive unique words
            if self.compute_unigram_entropy(text) < 2.0:  
                continue  # Skip documents that lack informative content

            # If the document passes all filters, add it to the filtered dataset
            filtered_data.append(sample)
        
        print(f"Original dataset size: {len(self.dataset)}")
        print(f"Filtered dataset size: {len(filtered_data)}")
        return filtered_data

if __name__ == "__main__":
    dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split='train')
    rule_filter = RuleBasedFilter(dataset, based_on=None)
    filtered_data = rule_filter.run()



