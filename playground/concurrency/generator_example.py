"""

yield https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
yield is not as magical this answer suggests. When you call a function that contains a yield statement anywhere, you get
a generator object, but no code runs. Then each time you extract an object from the generator, Python executes
code in the function until it comes to a yield statement, then pauses and delivers the object. When you extract another
object, Python resumes just after the yield and continues until it reaches another yield (often the same one, but one
iteration later). This continues until the function runs off the end, at which point the generator is deemed exhausted.

1. https://realpython.com/introduction-to-python-generators/
2. https://wiki.python.org/moin/Generators
3.
"""

def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

if __name__ == '__main__':
    # - calling a function with yield actually returns a generator
    gen = infinite_sequence()
    # - get a few values
    print(next(gen))
    print(next(gen))
    print(next(gen))
    for i in gen:
        print(f'next value of gen: {i}')
        if i > 10:
            break

    print('Success, done\a')
