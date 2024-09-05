# 2.1 Remove Dups: Write code to remove duplicates from an unsorted linked list.

from typing import Optional, Any

class Node:

    def __init__(self, val: Any):
        self.val = val
        self.next = None

class Link_List: 
    """
    Empty ll if head & tail points to None.
    tail always points to non-empty tail (otherwise it might point to None)
    head points to first node added
    empty node == just None
    """

    def __init__(self):
        self.head = None
        self.tail = self.head

    def append_end(self, val: Any) -> None:
        """mutates
        - if ll is empty then both head & tail point to None
        - if ll is added a val for first time, then both head & tail point to new node (head)
        - if added new node once ll is not empty, always move tail to point to this new node
        """
        new_node = Node(val)
        if self.head is None:
            assert self.tail is None
            self.head = new_node
            self.tail = self.head
            assert self.head is self.tail
        elif self.head is self.tail:
            self.tail = new_node
            self.head.next = self.tail
        else:
            prev_node = self.tail
            self.tail = new_node
            prev_node.next = self.tail

    def __repr__(self) -> str:
        str_rep: str = ''
        if self.head is None:
            assert self.tail is None
            assert str_rep == ''
            return str_rep  # should be ''
        else:
            prev_node: Node = self.head
            curr_node: Node = prev_node.next
            str_rep = f'{str_rep} {prev_node.val}_h'
            # invariant, anything at prev_node and before has been processed and added to string rep
            while curr_node is not None:
                str_rep = f'{str_rep} {curr_node.val}'
                prev_node = curr_node
                curr_node = curr_node.next
            str_rep = f'{str_rep}_t'
            return str_rep

def remove_dupilcates_with_tmp_buffer_ll(link_list: Link_List) -> None:
    """ 

    - store in a dict values seen
    - consider current node, & remembe prev node. If current node val has been seen remove point form current node to next & have prev point skip this node

    Note: mutates linked list.
    """
    assert isinstance(link_list, Link_List), f'Err: only dedups link lists, but got: {type(head)}'
    if link_list.head is None:
        assert link_list.tail is None
        return  # if empty link list nothing to do deduplicate
    elif link_list.head is link_list.tail:
        assert link_list.head.next is None
        return  # nothing to remove
    else: 
        prev_node: Node = link_list.head
        curr_node: Union[Node, None] = prev_node.next
        vals: dict[Any, int] = {prev_node.val: int}
        # invariant: anything at prev_node and before has been deduplicated
        while curr_node is not None:  
            curr_val = curr_node.val
            if curr_val in vals:
                # remove current node by skipping it, no need to change prev_node (remains the same, don't change it to removed node)
                prev_node.next = curr_node.next
                curr_node = curr_node.next
            else:
                # if current node is new, then add to vals & move both nodes together
                vals[curr_val] = 1
                prev_node = curr_node
                curr_node = curr_node.next


# def remove_dupilcates_no_tmp_buffer_ll(link_list: Link_List ) -> Link_List:
#     """ 

#     Note: mutates linked list.
#     """
#     pass

def remove_duplicates_test_():
    # empty -> empty
    print('\n-- empty -> empty')
    ll = Link_List()
    print(f'{ll=}')
    remove_dupilcates_with_tmp_buffer_ll(ll)
    print(f'{ll=}')
    # 1 -> 1
    print('\n-- 1 -> 1')
    ll.append_end(1)
    print(f'{ll=}')
    remove_dupilcates_with_tmp_buffer_ll(ll)
    print(f'{ll=}')
    # 1 1 -> 1
    print('\n-- 1 1 -> 1')
    ll = Link_List()
    ll.append_end(1)
    ll.append_end(1)
    print(f'{ll=}')
    remove_dupilcates_with_tmp_buffer_ll(ll)
    print(f'{ll=}')
    # 1 2 -> 1 2
    print('\n1 2 -> 1 2')
    ll = Link_List()
    ll.append_end(1)
    ll.append_end(2)
    print(f'{ll=}')
    remove_dupilcates_with_tmp_buffer_ll(ll)
    print(f'{ll=}')   
    # 1 1 2 2 3 -> 1 2 3
    print('\n1 1 2 2 3 -> 1 2 3')
    ll = Link_List()
    ll.append_end(1)
    ll.append_end(1)
    ll.append_end(2)
    ll.append_end(2)
    ll.append_end(3)
    print(f'{ll=}')
    remove_dupilcates_with_tmp_buffer_ll(ll)
    print(f'{ll=}')   
    # 1 2 3 4 3 4 3 1 2 1 1 1 4 4 0 -> 1 2 3 4 3 0
    print(f'\n-- 1 2 3 4 3 4 3 1 2 1 1 1 4 4 0 -> 1 2 3 4 3 0')
    ll = Link_List()
    ll.append_end(1)
    ll.append_end(2)
    ll.append_end(3)
    ll.append_end(4)
    ll.append_end(3)
    ll.append_end(4)
    ll.append_end(3)
    ll.append_end(1)
    ll.append_end(2)
    ll.append_end(1)
    ll.append_end(1)
    ll.append_end(1)
    ll.append_end(4)
    ll.append_end(4)
    ll.append_end(0)
    print(f'{ll=}')
    remove_dupilcates_with_tmp_buffer_ll(ll)
    print(f'{ll=}')  

# Return Kth to Last: Implement an algorithm to find the kth to last element of a singly linked list.

def return_kth_to_last(link_list: Link_List, kth_last: int) -> Any:
    """ 
    
    algorithm
    - search to extend start_n to end_n s.t. it's length k by traversing until the end until
        - if end_n.next === None and k > 1 ==> return None/Fail
        - halt if => k == 1 or end.next === None
    - now move both start and end refs until end.next is None
    """
    # assert kth_last >= 1, f'Err: we are indexing from 1, so 1st to last is the last element.'
    if kth_last < 1:
        return None
    if link_list.head is None:
        return None  # way to distiguish with geuinuine None could be having value being wrapped ina Val object (or type restrict the ll)
    elif link_list.tail is link_list.head:
        return link_list.head.val
    else:
        start_node: Node = link_list.head
        end_node = start_node
        counter_k: int = kth_last
        while counter_k > 1 and end_node.next is not None:
            counter_k -= 1
            end_node = end_node.next
        # 1 0 --> we wanted to keep searching but it's the end! kth element form end doesn't exist --> return None/Fail
        if counter_k > 1 and end_node.next is None:
            return None  # we can wrap values in Val() object to distinguish returning None vs Val(None)
        # 0 0 --> found kth element! we can catch this, bellow because it will see it's None and not enter loop and return current end_node.val
        # 0 1 --> continue searching for until the end!
        while end_node.next is not None:
            end_node = end_node.next
            start_node = start_node.next
        return start_node.val
            
def kth_to_last():
    # 
    print()
    ll = Link_List()
    ll.append_end('a3');ll.append_end('a2');ll.append_end('a1')
    print(f'{ll=}')
    kth = return_kth_to_last(ll, 3)
    assert kth == 'a3'
    kth = return_kth_to_last(ll, 2)
    assert kth == 'a2'
    kth = return_kth_to_last(ll, 1)
    assert kth == 'a1'
    kth = return_kth_to_last(ll, 4)
    assert kth == None
    kth = return_kth_to_last(ll, 5)
    assert kth == None
    kth = return_kth_to_last(ll, 0)
    assert kth == None

if __name__ == '__main__':
    # remove_duplicates_test_()
    kth_to_last()
    print(f'Done!\a')