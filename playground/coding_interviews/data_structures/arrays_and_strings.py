"""Chapter 1
"""
from collections import defaultdict

#  1.1 Is Unique: Implement an algorithm to determine if a string has all unique characters. What if you cannot use additional data structures?

def is_unique(string: str) -> bool:
    """
    1.1 Is Unique: Implement an algorithm to determine if a string has all unique characters. What if you cannot use additional data structures?

    Unique := means that all chars only appear once, so counter == 1 for all chars.
    not Unique := any char has a counter > 1, we can stop early once we detect this.

    How should we define ""? is it unique?
    conceptually, a string is unique if all chars are unique. So it computes a all(chars are unique). Thus, an all chars is unique fun is a large conjuction/and.
    So conceptually the loop starts with True and it's \and with True if invariant is still True i.e., current char num == 1. Once a repeated char
    appears we should \and the condition with False. So loop starts with True and string is empty it never ands anything and returns True.
    It's not unique only if there are no duplicates. Once there is a duplicate, then it returns false. So since the empty string has no duplicates
    return true.
    """
    num_times_char_appears: dict[str, int] = {}
    for char in string:
        if char not in num_times_char_appears:
            num_times_char_appears[char] = 1
        else:
            # char has appeared twice so we can return not unique already
            num_times_char_appears[char] += 1
            assert num_times_char_appears[char] > 1
            return False
    # assert all(num_times_char_appears[key] > 1 for key in num_times_char_appears.keys())
    return True

def is_unique_unit_test_():
    assert is_unique("") 
    assert is_unique("a") 
    assert is_unique("ab") 
    assert is_unique("abc"), f"String: {'abc'} has all unique characters"
    assert not is_unique("abca"), f"String: {'abca'} does not have all unique characters"
    # todo: with py Set?

# 

if __name__ == "__main__":
    is_unique_unit_test_()
    print('Done!\a')