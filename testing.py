import numpy as np
import random

random_list = ["asdf1", "asdf2", "asdf3", "asdf4-0", "asdf5-0"]
random_labels = [1, 1, 1, 0, 0]

random_list2 = ["asdf6", "asdf7", "asdf8", "asdf9-0", "asdf10-0"]
random_labels2 = [1, 1, 1, 0, 0]

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

print(random_list_labels)

