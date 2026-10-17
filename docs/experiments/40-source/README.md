# 第 40 章证据和代码

`published-files.json` 中路径相对于 `docs/experiments`；它记录从原发布包逐字节核验并复制的文件。V117 原 receipt 采用无损 gzip，解压 SHA 为 `bbc2067380d4ecac328427411a92853c8587a0a16ea11cf12bb8da2b370491e5`。摘要没有完整设备 tensor，因此不能离线重算 GPU 值。

V117/V120 的源文件保留当时路径与依赖 hash；它们是实验快照，直接离机运行需重建绑定。完整 candidate116 代码见 [第 39 章固定版本](../../code-versions/39-broadcast-m1/README.md)。V120 两次资源中止和 V122 小缓存诊断没有投机性能点；资源只发布白名单摘要，原进程 argv 不在包内。

实际完成后另有 [V117 独立结果审查](v117-actual-review/REVIEW.md)，区别于运行前的源码审查。此审查解析保存结果，没有重新执行 GPU；原本机路径和 SHA 保留。
