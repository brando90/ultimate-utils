# Given an array arr[] of size N and an integer K, the task is to find the count of distinct pairs in the array whose sum is equal to K.

def count_distinct_pairs_in_sum_equals_k(numbers: list[int], target: int):
    """ Count number of distinct pairs that sum to target. 

    Input:  = [5, 6, 5, 7, 7, 8], target = 13 
    Output: 2
    pairs, target=13 = (6, 7)[1th, 3rd], (6, 7)[1th, 4th], (5, 8)[0th, 5th]

    pair is distinct if at least 1 element is different.
    """
    print(f'{numbers=}')
    print(f'{target=}')
    pairs_sum_eq_target: dict[tuple, bool] = {}  # if pair in dict then it's been seen before then -> True (unique) else if already seen before -> False (not unique, distinct)
    # for idx1, num1 in enumerate(numbers):
    for idx1 in range(len(numbers)):
        num1 = numbers[idx1]
        print(f'\n{idx1=}')
        # for idx2, num2 in enumerate(numbers, start=idx1+1):
        range
        # for num2 in range(start=idx1+1, stop=len(numbers)):
        for idx2 in range(idx1+1, len(numbers)):
            print(f'{idx2=}')
            num2 = numbers[idx2]
            # print(f'{idx2=}')
            # assert idx2 < len(numbers), f'Err: {idx2=} but max len of list is {len(numbers)=}'
            pair: tuple[int, int] = (num1, num2)
            sum_pair: int = sum(pair)
            if sum_pair == target:
                if pair in pairs_sum_eq_target:
                    pairs_sum_eq_target[pair] = True  # not been seen before so unique/distinct
                else:
                    pairs_sum_eq_target[pair] = False  # been seen before so not unique/distinct
    print(f'{pairs_sum_eq_target=}')
    # now we want to know how many things have not been seen before (distinct/unique)
    count: int = 0
    for distinct in pairs_sum_eq_target.values():
        if distinct:
            count += 1
    print(f'{count=}')
    return count

def distinct_pairs_test_():
    ## test distinct pairs
    assert (1, 1) == (1, 1), 'Err: (1, 1) should be equal to (1, 1) (not distinct).'
    assert (1, 2) != (2, 1), 'Err: (1, 2) is distinct from (2, 1) so they should not be equal pairs'
    assert (2, 1) != (1, 2), 'Err: (2, 1) is distinct from (1, 2 so they should not be equal pairs)'
    ## test
    # [1, 1] target 2
    # assert count_distinct_pairs_in_sum_equals_k([1, 1], 2) == 1
    # [5, 6, 5, 7, 7, 8], target = 13
    # Pairs with sum target = 13 are [ (arr[0], arr[5]), (arr[1], arr[3]), (arr[1], arr[4]) }, i.e. {(5, 8), (6, 7), (6, 7)]. 
    # Therefore, distinct pairs with sum target = 13 are { (arr[0], arr[5]), (arr[1], arr[3]) }. 
    # Therefore, the required output is 2.
    assert count_distinct_pairs_in_sum_equals_k([5, 6, 5, 7, 7, 8], 13) == 2

if __name__ == '__main__':
    distinct_pairs_test_()
    print(f'Done!\n')