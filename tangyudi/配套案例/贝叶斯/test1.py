alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion


word = 'xiaoming'
n = len(word)

# print(set(word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)))

# print(list(word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet))

# print(list(word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet))

print(set([word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet]))

print("aaaa")