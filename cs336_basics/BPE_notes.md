# Byte-level BPE：2-minute 级方案 vs 朴素实现（技术总结）

> 目标：在 TinyStories 上训练 byte-level BPE（vocab_size≈10k），满足特殊 token `<|endoftext|>` 约束，并把训练时间压到分钟级。

---

## 朴素版（你之前的方法）在做什么

### 流程
1. **Pre-tokenization**：regex 切分文本 → 构建  
   `freq_table: dict[tuple[int, ...], int]`
2. 每一轮 merge（直到 vocab_size）：
   - **全量扫描** `freq_table`，统计所有相邻 pair 的频次 `count_table`
   - 取最高频 pair（按规则 tie-break）
   - **再次全量扫描** `freq_table`，把所有出现该 pair 的位置替换成新 token id，构造 `new_freq_table`

### 本质复杂度（为什么慢）
- 每轮 merge 都要扫 **整个语料的 token 序列总量**（至少一遍计数 + 一遍替换）
- 10k merges ≈ **全语料扫描 10k 次**：即使并行部分计数，Python dict/tuple 分配与重写依旧是大头  
  → 很难做到分钟级

---

## 2-minute 级方案（现在的方法）在做什么

核心：**把“每轮全量重算/重写”改成“初始化一次 + 每轮只增量更新受影响部分”**，并把多进程放在最赚的阶段（pretokenization）。

---

## 关键不同点与优化点

### 1) 多进程用在 pretokenization（不是每轮 merge）
- **现在**：读取文件时按 `<|endoftext|>` 切文档 → 批量分发给 worker 做 `regex.finditer` → worker 返回局部 `Counter` → 主进程合并
- **优化原因**：regex 是一次性、CPU 重活、天然可并行；每轮 merge 开进程池的启动/IPC/合并成本往往吃掉收益

---

### 2) pretoken key 用 `bytes`（避免早期产生海量 tuple）
- **现在**：pretokenization 输出 `Counter[bytes]`（如 `b" hello"`）
- 之后才把 bytes 转成 `tuple(b)` 进入 merges 逻辑
- **优化原因**：`tuple(tok.encode())` 会制造大量 Python tuple 对象，分配/GC 成本高；`bytes` 更紧凑、hash 更快、内存更省

---

### 3) pair 统计“初始化一次”+ 倒排索引定位受影响词
初始化一次构建：
- `pair_counts[pair] = 总频次`
- `pair_to_words[pair] = {word_id...}`（倒排索引：哪些 word 含该 pair）

每轮 merge 仅更新：
- `affected = pair_to_words[pair]` 中的 word
- 对每个受影响 word：
  - 在该 word 内进行 pair merge
  - 重新计算该 word 的相邻 pairs（仅局部）
  - **增量更新** `pair_counts` 与 `pair_to_words`

**优化原因**：大多数 word 不包含当前 pair → 不需要改；每轮成本从“全语料”变成“只更新含该 pair 的子集”。

---

### 4) heap 维护“当前最大 pair” + lazy 更新（避免每轮全表 max）
- **现在**：用 heap 存候选 `( -count, tie_break_key..., pair )`
- pop 时做 **lazy validation**：
  - heap 里可能有过期条目（旧 count），验证不匹配就丢弃继续 pop
- 当某 pair count 更新时，直接 push 新条目（旧条目留给 lazy 丢弃）

**优化原因**：避免每轮对整个 `pair_counts` 扫一遍找 max；heap 的 `push/pop` 是 `O(log P)`，P 是 pair 种类数，通常远小于语料 token 总量。

---

### 5) special token `<|endoftext|>` 的处理符合 hint (a)(b)
- (a) `<|endoftext|>` 是 **文档边界**：按它切分文档并独立 pretokenize，禁止跨文档统计 pair
- (b) `<|endoftext|>` 作为 **special token**：训练时不进入 regex、不参与 merge，最终单独加入 vocab

**优化原因**：防止学出 `<|`、`|e`、`text|>` 等垃圾 token；减少无意义 pair 与候选空间；也满足测试/语义约束。

---

## 一句话结论
- **朴素版**：每轮 merge 都 **全量重算 + 全量重写**（≈ `O(K * |corpus|)`）
- **2-minute 级方案**：并行预分词一次 + 初始化全局 pair 统计一次 + 每轮只更新受影响词 + heap 维护最大 pair  
  （≈ `O(|corpus| + K * |affected| log P)`）

---

## 直觉小例子（为什么增量更新快）
如果 100 万个 pretoken 里只有 1 万个包含当前 pair：
- 朴素法：每轮扫 100 万个
- 增量法：每轮只更新这 1 万个（其余 99 万个完全不动）

语料越大、merge 越多，收益越明显。