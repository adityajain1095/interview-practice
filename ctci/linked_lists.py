#create linked list

class Node(object):

    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class linkedList(object):

    def __init__(self, head=None):
        self.head = head
        self.tail = head
        self.size = 0
        if self.head:
            self.size += 1

    def insert(self,data):
        if self.size:
            self.size += 1
            node = Node(data)
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
            return node
        else:
            self.size += 1
            self.head = Node(data)
            self.tail = self.head
            return self.head

    def search(self, data):
        if self.size:
            node = self.head
            while node != None:
                if node.data == data:
                    return node
                node = node.next
            return None

    def remove(self, data):
        if self.size:
            node = self.search(data)
            if node:
                self.size -= 1
                if self.head == node:
                    self.head = node.next
                elif self.tail == node:
                    self.tail = self.tail.prev
                else:
                    node.prev.next = node.next
            return node

    def isEmpty(self):
        if self.size:
            return False
        return True

ll = linkedList()

ll.insert(1)
ll.insert(1)
ll.insert(5)
ll.insert(3)
ll.insert(2)
ll.insert(4)
ll.insert(6)
ll.insert(7)
ll.insert(4)

def two_one(ll):
    pointerOne = ll.head.next
    pointerTwo = ll.head
    viewed = set()
    while pointerOne:
        viewed.add(pointerTwo.data)
        temp = pointerOne.data
        pointerTwo = pointerTwo.next
        pointerOne = pointerOne.next
        if temp in viewed:
            ll.remove(temp)
    return ll.head.data

# two_one(ll)
# node = ll.head
# while node:
#     print(node.data)
#     node = node.next
