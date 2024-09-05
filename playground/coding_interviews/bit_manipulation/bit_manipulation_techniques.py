"""
When doing these make sure you remember bit facts/tricks

todo: implement all https://www.techinterviewhandbook.org/algorithms/binary/ 
"""

def get_kth_bit(num: int, k_idx: int) -> int:
    """ Returns kth bit from number. k_idx is zero indexed.
    
    Algorithm:
    - get a mask with a 1 in the location you want to test
    - clear out all other bits
    - if the bit was a 1 then the resulting number won't be zero
    - if the bit was a zero then the resulting number will be zero
    """
    # mask for kth bit
    mask: int = 1 << k_idx
    # clear all bits except kth e.g., 1010 & 0010 -> 0010 
    result: int = num & mask
    if result == 0:
        # result == 0 then the kth bit was zero ==> return zero
        return 0  # int(result != 0)
    else:
        # result != 0 then the kth bit was not zero ==> return 1
        return 1  # int(resul != 0)

def set_kth_bit(num: int, k_idx: int) -> int:
    """Sets the kth bit in the given number. 
    
    At position k you want it do the follow ops:
    1 -> 1
    0 -> 1
    rest bits remain untouched.
    """
    return num | (1 << k_idx)

def clear_bit(num: int, k_idx: int) -> int: 
    """clears kth bit.
    
    Algorithm:
    at k we want
    1 -> 0
    0 -> 0
    remaining bits unchanged
    so and zero at k and 1's at rest since 1 is identity op for and.
    """
    return num & ~(1 << k_idx)

def clear_bits_msb_through_k(num: int, k: int) -> int:
    """ clear all bits from the most significant bit to the kth bit. 
    
    Algorithm:
    from msb (front) to k (inclusive) we want to set all bits to zero.
    - create bit with 1 at kth position
    - subtract one so to create zeros infront and 1s behind (due to carry on in debustrtion)
    - then and the two (since 0 destroys in and and 1 is the identity in and)
    """
    mask: int = 1<<k
    mask2: int = mask - 1
    result: int = num & mask2
    return result

def clear_bits_k_through_0(num: int, k: int):
    """ inclusive"""
    mask: int = (-1 << k)
    return num & mask

def update_bit_to_value_v(num: int, k: int):
    pass

def bit_test_():
    ## get_kth_bit
    # test 1101 & 0001 --> 0
    assert get_kth_bit(13, 0) == 1
    # test 1101 & 0010 --> 1
    assert get_kth_bit(13, 1) == 0
    ## set bit test
    # 0000 -> 0001
    assert set_kth_bit(0, 0) == 1
    # 0001 -> 0101
    assert set_kth_bit(1, 2) == 5
    ## clear bit test
    # 1010 -> 1000
    assert clear_bit(10, 1) == 8
    ## clear bits from msb to i (inclusive)
    # 11101 -> 00101  
    assert clear_bits_msb_through_k(29, 3) == 5
    # 11111 -> 00000 
    assert clear_bits_msb_through_k(29, 0) == 0
    ## clear bits from k (inclusive) to beginning (0)
    # 1111 -> 1110
    assert clear_bits_k_through_0(15, 1) == 14
    # 1111 -> 1100
    assert clear_bits_k_through_0(15, 2) == 12

if __name__ == '__main__':
    bit_test_()
    print('Done!\a')