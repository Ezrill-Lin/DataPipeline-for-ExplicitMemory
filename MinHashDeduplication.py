from datasketch import MinHash, MinHashLSH
import string
from datasets import load_dataset
from typing import List, Dict, Any

class Deduplicate():
    def __init__(self, dataset, based_on: str, threshold: float=0.8, num_perm: int=128) -> None:
        '''
        Perform the MinHash filtering on question part of each document. Since 
        the categorical name for each dataset's question part is different, we
        need to specify the name of the question part in "based_on" argument

        dataset (Hugging-Face dataset): dataset found on Hugging-Face
        based_on (str): the name of question part in dataset
                        based on which we perform the Minhash filtering
        threshold (float): the Jaccard similarity threshold, exceed which we consider 
                            two pieces of data is duplicated
        num_perm (int): arg required to create `MinHash` object, the number of Hash 
                        functions applied to each document to perform MinHash filtering
        '''
        self.dataset = dataset
        self.based_on = based_on # based on which category are we duduplicating the dataset
        self.threshold = threshold
        self.num_perm = num_perm

    # Function to preprocess text (tokenization into words)
    def preprocess_text(self, text: str) -> set:
        text = text.lower().translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        return set(text.split())  

    # Function to create a MinHash signature for a text sample
    def create_minhash(self, text: str) -> MinHash:
        minhash = MinHash(num_perm=self.num_perm)
        for word in self.preprocess_text(text):
            minhash.update(word.encode('utf8'))
        return minhash
    
    def run(self) -> List[Dict[str, Any]]:
        # Define LSH to efficiently group similar MinHashes
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        minhashes = {}
        for i, sample in enumerate(self.dataset):
            # some dataset may contain lists of dictionaries in sub-categories
            try: 
                text = sample[self.based_on]
                minhash = self.create_minhash(text)
            except: 
                text = sample[self.based_on][0]['input']
                minhash = self.create_minhash(text)
            
            lsh.insert(str(i), minhash)  # Insert into LSH
            minhashes[str(i)] = minhash

        # Find duplicates
        duplicates = set()
        for i, sample in enumerate(self.dataset):
            key = str(i)
            if key in duplicates:
                continue
            similar_items = lsh.query(minhashes[key])  # Get similar items
            if len(similar_items) > 1:
                duplicates.update(similar_items[1:])  # Mark duplicates

        # Create deduplicated dataset
        deduplicated_ds = [sample for i, sample in enumerate(self.dataset) if str(i) not in duplicates]

        # Display results
        print(f"Original dataset size: {len(self.dataset)}")
        print(f"Deduplicated dataset size: {len(deduplicated_ds)}")

        return deduplicated_ds



if __name__== "__main__":
    dataset = load_dataset("LDJnr/Capybara", split='train')
    ds_duplicate = Deduplicate(dataset=dataset, based_on='conversation')
    ds_duplicate.run()