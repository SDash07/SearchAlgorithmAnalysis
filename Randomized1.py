import random
import time
import copy
from typing import Any, Iterable, Optional, List

# --------------------------
# Part 1: Selection Algorithms
# --------------------------

def insertion_sort(arr: List[Any], left: int, right: int):
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def partition(arr: List[Any], left: int, right: int, pivot_index: int) -> int:
    pivot_value = arr[pivot_index]
    arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
    store_index = left
    for i in range(left, right):
        if arr[i] < pivot_value:
            arr[store_index], arr[i] = arr[i], arr[store_index]
            store_index += 1
    arr[store_index], arr[right] = arr[right], arr[store_index]
    return store_index

def select_pivot_median_of_medians(arr: List[Any], left: int, right: int) -> int:
    n = right - left + 1
    if n <= 5:
        insertion_sort(arr, left, right)
        return left + n // 2

    medians = []
    i = left
    while i <= right:
        group_right = min(i + 4, right)
        insertion_sort(arr, i, group_right)
        median_idx = i + (group_right - i) // 2
        medians.append(arr[median_idx])
        i += 5

    # Recursively find median of medians
    mom = deterministic_select(medians, 0, len(medians) - 1, len(medians) // 2)
    # Find index of mom in original segment
    for j in range(left, right + 1):
        if arr[j] == mom:
            return j
    return left  # fallback

def deterministic_select(arr: List[Any], left: int, right: int, k: int) -> Any:
    if left == right:
        return arr[left]
    pivot_index = select_pivot_median_of_medians(arr, left, right)
    pivot_index = partition(arr, left, right, pivot_index)
    rank = pivot_index - left  # zero-based rank
    if k == rank:
        return arr[pivot_index]
    elif k < rank:
        return deterministic_select(arr, left, pivot_index - 1, k)
    else:
        return deterministic_select(arr, pivot_index + 1, right, k - rank - 1)

def randomized_partition(arr: List[Any], left: int, right: int) -> int:
    pivot_index = random.randint(left, right)
    return partition(arr, left, right, pivot_index)

def randomized_select(arr: List[Any], left: int, right: int, k: int) -> Any:
    if left == right:
        return arr[left]
    pivot_index = randomized_partition(arr, left, right)
    rank = pivot_index - left
    if k == rank:
        return arr[pivot_index]
    elif k < rank:
        return randomized_select(arr, left, pivot_index - 1, k)
    else:
        return randomized_select(arr, pivot_index + 1, right, k - rank - 1)

# --------------------------
# Part 2: Elementary Data Structures
# --------------------------

class SimpleArray:
    def __init__(self):
        self._data: List[Any] = []

    def access(self, index: int) -> Any:
        return self._data[index]

    def insert(self, index: int, value: Any):
        self._data.insert(index, value)

    def delete(self, index: int):
        del self._data[index]

    def append(self, value: Any):
        self._data.append(value)

    def __len__(self):
        return len(self._data)

    def to_list(self):
        return list(self._data)

class Stack:
    def __init__(self):
        self._data: List[Any] = []

    def push(self, item: Any):
        self._data.append(item)

    def pop(self) -> Any:
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self) -> Any:
        if not self._data:
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

class Queue:
    def __init__(self):
        self._data: List[Any] = []
        self._head: int = 0

    def enqueue(self, item: Any):
        self._data.append(item)

    def dequeue(self) -> Any:
        if self._head >= len(self._data):
            raise IndexError("dequeue from empty queue")
        item = self._data[self._head]
        self._head += 1
        if self._head > 50 and self._head * 2 > len(self._data):
            self._data = self._data[self._head:]
            self._head = 0
        return item

    def is_empty(self) -> bool:
        return self._head >= len(self._data)

    def __len__(self):
        return len(self._data) - self._head

class Node:
    def __init__(self, value: Any):
        self.value = value
        self.next: Optional['Node'] = None

class SinglyLinkedList:
    def __init__(self):
        self.head: Optional[Node] = None

    def insert_front(self, value: Any):
        node = Node(value)
        node.next = self.head
        self.head = node

    def insert_back(self, value: Any):
        node = Node(value)
        if not self.head:
            self.head = node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = node

    def delete(self, value: Any) -> bool:
        prev = None
        curr = self.head
        while curr:
            if curr.value == value:
                if prev:
                    prev.next = curr.next
                else:
                    self.head = curr.next
                return True
            prev = curr
            curr = curr.next
        return False

    def traverse(self) -> Iterable[Any]:
        curr = self.head
        while curr:
            yield curr.value
            curr = curr.next

    def to_list(self):
        return list(self.traverse())

# --------------------------
# Driver / Demo
# --------------------------

def time_selector(func, arr: List[int], k: int):
    a = copy.deepcopy(arr)
    start = time.perf_counter()
    result = func(a, 0, len(a) - 1, k)
    end = time.perf_counter()
    return result, end - start

def basic_self_test():
    print("=== Selection Algorithm Tests ===")
    arrays = [
        [5],
        [3, 1, 2, 4, 5],
        list(range(20)),
        list(range(20))[::-1],
        [random.randint(0, 5) for _ in range(30)],
    ]
    for arr in arrays:
        n = len(arr)
        for k in [0, n // 2, n - 1]:
            det = deterministic_select(arr.copy(), 0, n - 1, k)
            rand = randomized_select(arr.copy(), 0, n - 1, k)
            expected = sorted(arr)[k]
            assert det == expected, f"Deterministic failed on {arr} k={k}"
            assert rand == expected, f"Randomized failed on {arr} k={k}"
    print("Selection basic tests passed.")

    print("\n=== Data Structures Tests ===")
    sa = SimpleArray()
    sa.append(10)
    sa.append(20)
    sa.insert(1, 15)
    assert sa.access(1) == 15
    sa.delete(1)
    assert sa.to_list() == [10, 20]

    stack = Stack()
    stack.push(1)
    stack.push(2)
    assert stack.peek() == 2
    assert stack.pop() == 2
    assert not stack.is_empty()
    assert stack.pop() == 1

    queue = Queue()
    queue.enqueue("a")
    queue.enqueue("b")
    assert queue.dequeue() == "a"
    assert queue.dequeue() == "b"
    try:
        _ = queue.dequeue()
        assert False, "Expected dequeue from empty"
    except IndexError:
        pass

    ll = SinglyLinkedList()
    ll.insert_back(1)
    ll.insert_front(0)
    ll.insert_back(2)
    assert ll.to_list() == [0, 1, 2]
    assert ll.delete(1) is True
    assert ll.to_list() == [0, 2]
    assert ll.delete(99) is False

    print("Data structure basic tests passed.")

def interactive_demo():
    print("\n=== Interactive selection demo ===")
    raw = input("Enter list of integers separated by spaces (or 'r' for random): ").strip()
    if raw.lower() == 'r':
        n = int(input("Size of random array? "))
        arr = [random.randint(0, n * 10) for _ in range(n)]
    else:
        arr = list(map(int, raw.split()))
    if not arr:
        print("Empty array; nothing to select.")
        return
    print(f"Array (first 100 shown): {arr[:100]}{'...' if len(arr)>100 else ''}")
    k = int(input(f"Enter k (0-based, 0..{len(arr)-1}): "))
    if k < 0 or k >= len(arr):
        print("Invalid k.")
        return
    det_res, det_time = time_selector(deterministic_select, arr, k)
    rand_res, rand_time = time_selector(randomized_select, arr, k)
    expected = sorted(arr)[k]
    print(f"\nExpected {k}-th smallest (by sorting): {expected}")
    print(f"Deterministic select result: {det_res}, time: {det_time:.6f}s")
    print(f"Randomized select result:   {rand_res}, time: {rand_time:.6f}s")

def benchmark_once():
    print("\n=== Simple benchmark on random data ===")
    n = 200000
    arr = [random.randint(0, 1000000) for _ in range(n)]
    k = n // 2
    det_res, det_time = time_selector(deterministic_select, arr, k)
    rand_res, rand_time = time_selector(randomized_select, arr, k)
    print(f"Array size: {n}, median k={k}")
    print(f"Deterministic: {det_res} in {det_time:.4f}s")
    print(f"Randomized:    {rand_res} in {rand_time:.4f}s")
    assert det_res == rand_res, "Mismatch in medians"

if __name__ == "__main__":
    basic_self_test()
    benchmark_once()
    # Uncomment to enable interactive prompt
    # interactive_demo()
