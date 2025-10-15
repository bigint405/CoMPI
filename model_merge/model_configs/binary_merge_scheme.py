from model_merge.model_configs.base_merge_scheme import BaseMergeScheme

# 融合方案每次分为左右两层，通过一个二进制数 code 记录每个方案所属层级
class BinaryMergeScheme(BaseMergeScheme):
    code: int = 0

    def isSubScheme(self, other):
        a = self.code
        b = other.code
        if a <= b:
            return False
        n = b.bit_length()
        a_prefix = a >> (a.bit_length() - n)
        return b == a_prefix

class SplitScheme():
    start: int
    end: int
    spliter: int

    def __init__(self, start: int, end: int, spliter: int):
        self.start = start
        self.end = end
        self.spliter = spliter

class BinaryMergeSchemesManager():
    schemes: dict[int, list[BinaryMergeScheme]] # code -> [scheme]
    spliters: dict[int, SplitScheme] # code -> spliter
    sub_code: dict[int, set[int]] # code -> [codes]

    def __init__(self, schemes:list[BinaryMergeScheme], spliters: dict[int, SplitScheme]):
        self.schemes = dict()
        self.sub_code = dict()
        for s in schemes:
            self.add_scheme(s)
        self.spliters = spliters

    def add_scheme(self, scheme:BinaryMergeScheme):
        if scheme.code not in self.sub_code:
            self.sub_code[scheme.code] = set()
        if scheme.code not in self.schemes:
            self.schemes[scheme.code] = []
        self.schemes[scheme.code].append(scheme)
        for sl in self.schemes.values():
            for s in sl:
                if scheme.isSubScheme(s):
                    self.sub_code[s.code].add(scheme.code)
    
    def get_all_schemes(self):
        s = []
        for ss in self.schemes.values():
            s.extend(ss)
        return s
