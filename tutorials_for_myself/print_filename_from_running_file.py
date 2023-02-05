# print full path from python file from within running python file
print(f'{__file__=}')

# print filename from python file from within running python file
import os

print(os.path.basename(__file__))
