# 第34章冻结文本归档

[正文](../34-native-ba-gates-prework.md)只记录V76localnumeric、V82fullqual、V86B-only screen三个actual runs；[provenance](provenance.json)保留版本/资格/计时边界。

- [source-map](source-map.json)、[dependency-map](dependency-map.json)：精确原路径/长度/SHA；支持跨章`tar.gz::成员`，旧大gold不重复收录，必要来源按精确成员引用。
- [evidence-index](evidence-index.json)与[evidence.tar.gz](evidence.tar.gz)：127份原始文本；确定性gzip/tar，三次owner/release和资源原记录原字节保存。
- candidate82原29py及V76/82/86原型文本按SHA精确归档/去重；源码PENDING/CPU_READY状态是历史准备文字，以actual receipts为准。
- [82独立review](reviews/v82-independent-results-review-build1/REVIEW.md)及[CPU verifier](verify_archive.py)：720局部、12,276双gold、792终态与计时/释放核算。
- [external-fixture-identities](external-fixture-identities.json)：tensor/native/runtime仅身份；无payload。离机只做文本/哈希/统计，`--originals`另需本机原件。
- [verification](verification.json)、[verification-originals](verification-originals.json)、[manifest](manifest.json)：封存验证结果和全文件SHA。

V76 genericX与历史u/S不是连贯整模输入；所读oMLX Q35分支仍分开gates/recurrence，Qwen4 BF16 gates未复制。V82/V86仅源码声明14个buffer，实际逐参数反射NOT_OBSERVED。V86rawt1afterseed，无HTTP/配对/采用结论；candidate82 defaultOFF。

MLX MIT、Lithos/oMLX Apache2等完整原许可文本按原字节保留或引用；此归档不是独立GPU执行包。
