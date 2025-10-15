class UnionFind:
    def __init__(self, n):
        """初始化 n 个元素，编号从 0 到 n-1"""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.set_size = [1] * n  # 每个集合初始大小为 1

    def find(self, x):
        """查找 x 的根节点，带路径压缩"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        """合并 x 和 y 所在的集合"""
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False  # 已经在同一个集合中

        # 按秩合并
        if self.rank[x_root] < self.rank[y_root]:
            x_root, y_root = y_root, x_root  # 保证 x_root 的秩不小
        self.parent[y_root] = x_root
        self.set_size[x_root] += self.set_size[y_root]
        if self.rank[x_root] == self.rank[y_root]:
            self.rank[x_root] += 1
        return True

    def connected(self, x, y):
        """判断 x 和 y 是否在同一个集合"""
        return self.find(x) == self.find(y)

    def size(self, x):
        """返回 x 所在集合的大小"""
        return self.set_size[self.find(x)]
