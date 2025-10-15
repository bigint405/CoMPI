package main

import (
	"math/rand"
	"time"
)

// RandomRoundRobinSet 结构体
type RandomRoundRobinSet struct {
	values []string       // 存储所有键
	index  map[string]int // 记录键在 keys 切片中的索引
	r      *rand.Rand
}

// 创建 RoundRobinSet
func NewRRRSet() *RandomRoundRobinSet {
	return &RandomRoundRobinSet{
		values: []string{},
		index:  make(map[string]int),
		r:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// 插入值
func (rm *RandomRoundRobinSet) Insert(value string) {
	if _, exists := rm.index[value]; !exists {
		rm.values = append(rm.values, value) // 添加到 keys 列表
		rm.index[value] = len(rm.values) - 1 // 记录 key 在 keys 中的索引
	}
}

// 删除值
func (rm *RandomRoundRobinSet) Delete(value string) {
	idx, exists := rm.index[value]
	if !exists {
		return
	}
	lastKey := rm.values[len(rm.values)-1] // 取最后一个元素的 key

	// 用最后一个元素覆盖要删除的元素，然后更新索引
	rm.values[idx] = lastKey
	rm.index[lastKey] = idx

	// 删除最后一个元素
	rm.values = rm.values[:len(rm.values)-1]
	delete(rm.index, value)
}

// 随机id开始的轮询遍历
func (rm *RandomRoundRobinSet) Iter(f func(string) bool) {
	if len(rm.values) == 0 {
		return
	}
	endIdx := rm.r.Int() % len(rm.values)
	for i := (endIdx + 1) % len(rm.values); f(rm.values[i]) && i != endIdx; i = (i + 1) % len(rm.values) {
	}
}
