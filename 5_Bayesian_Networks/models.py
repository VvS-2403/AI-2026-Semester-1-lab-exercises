import random
from collections import defaultdict

data = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]

def preprocess(dataset):
    return [["<START>"] + sentence.lower().split() + ["<END>"] for sentence in dataset]

tokenized_data = preprocess(data)

# First Order Model
class FirstOrderModel:
    def __init__(self, data):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.probabilities = defaultdict(dict)
        self.train(data)
        
    def train(self, data):
        for sentence in data:
            for i in range(len(sentence) - 1):
                curr_word = sentence[i]
                next_word = sentence[i+1]
                self.transitions[curr_word][next_word] += 1
                
        for curr_word, next_words in self.transitions.items():
            total = sum(next_words.values())
            for next_word, count in next_words.items():
                self.probabilities[curr_word][next_word] = count / total
                
    def check_normalization(self):
        print("First Order Normalization Check:")
        for word, next_words in self.probabilities.items():
            total = sum(next_words.values())
            print(f"P(* | {word:10}) = {total}")

    def generate(self, mode="sample"):
        sentence = ["<START>"]
        while sentence[-1] != "<END>":
            curr_word = sentence[-1]
            if curr_word not in self.probabilities:
                break
            
            next_words = self.probabilities[curr_word]
            if mode == "greedy":
                # To break ties predictably in greedy mode, sort by word
                best_word = max(sorted(next_words.items()), key=lambda x: x[1])[0]
                sentence.append(best_word)
            else: # sample
                words = list(next_words.keys())
                probs = list(next_words.values())
                chosen = random.choices(words, weights=probs, k=1)[0]
                sentence.append(chosen)
                
        return " ".join(sentence)

# Second Order Model
class SecondOrderModel:
    def __init__(self, data):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.probabilities = defaultdict(dict)
        self.train(data)
        
    def train(self, data):
        for sentence in data:
            for i in range(len(sentence) - 2):
                w1 = sentence[i]
                w2 = sentence[i+1]
                w3 = sentence[i+2]
                self.transitions[(w1, w2)][w3] += 1
                
        for context, next_words in self.transitions.items():
            total = sum(next_words.values())
            for next_word, count in next_words.items():
                self.probabilities[context][next_word] = count / total
                
    def check_normalization(self):
        print("Second Order Normalization Check:")
        for context, next_words in self.probabilities.items():
            total = sum(next_words.values())
            print(f"P(* | {str(context):25}) = {total}")

    def generate(self, mode="sample"):
        sentence = ["<START>", "the"]
        while sentence[-1] != "<END>":
            context = (sentence[-2], sentence[-1])
            if context not in self.probabilities:
                break
            
            next_words = self.probabilities[context]
            if mode == "greedy":
                best_word = max(sorted(next_words.items()), key=lambda x: x[1])[0]
                sentence.append(best_word)
            else:
                words = list(next_words.keys())
                probs = list(next_words.values())
                chosen = random.choices(words, weights=probs, k=1)[0]
                sentence.append(chosen)
                
        return " ".join(sentence)

if __name__ == "__main__":
    random.seed(42)
    m1 = FirstOrderModel(tokenized_data)
    m1.check_normalization()
    
    print("\n--- Conditional Probabilities for specific words (First-Order) ---")
    words = ["the", "cat", "dog", "sat", "ran"]
    for w in words:
        print(f"P(next | {w:5}): {m1.probabilities.get(w, {})}")
        
    print("\n--- Generating First-Order (Greedy) ---")
    for _ in range(5):
        print(m1.generate("greedy"))
        
    print("\n--- Generating First-Order (Sample) ---")
    for _ in range(5):
        print(m1.generate("sample"))

    print("\n============================================\n")

    m2 = SecondOrderModel(tokenized_data)
    m2.check_normalization()
    
    print("\n--- Generating Second-Order (Greedy) ---")
    for _ in range(5):
        print(m2.generate("greedy"))
        
    print("\n--- Generating Second-Order (Sample) ---")
    for _ in range(5):
        print(m2.generate("sample"))
