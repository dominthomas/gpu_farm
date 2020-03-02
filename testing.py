import numpy as np
import random

random_list = ["asdf1", "asdf2", "asdf3", "asdf4", "asdf5"]
random_labels = [1, 1, 1, 1, 1]

random_list2 = ["bob1", "bob2", "bob3", "bob4", "bob5"]
random_labels2 = [0, 0, 0, 0, 0]

random.Random(129).shuffle(random_list)
random.Random(129).shuffle(random_labels)
random.Random(129).shuffle(random_list2)
random.Random(129).shuffle(random_labels2)

print(random_list)
print(random_labels)
print(random_list2)
print(random_labels2)

random_list_dictionary = {'train': random_list, 'validation': random_list2}
print(random_list_dictionary)

random_list_labels = {}
for item in random_list:
    random_list_labels[item] = 1


asdf = random_labels + random_labels2

random.Random(129).shuffle(asdf)
print(asdf)

