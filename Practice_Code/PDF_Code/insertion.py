# Write a Python program to insertion at the beginning in OrderedDict.
from collections import OrderedDict

Ordered_dict = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
new_item = ('d', 4)
new_ordered_dict = OrderedDict([new_item])

new_ordered_dict.update(Ordered_dict)
print(new_ordered_dict)