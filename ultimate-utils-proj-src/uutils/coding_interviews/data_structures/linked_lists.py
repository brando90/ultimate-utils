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

if __name__ == '__main__':
    remove_duplicates_test_()
    print(f'Done!\a')