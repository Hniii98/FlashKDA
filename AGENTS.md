# Topic7：C32 实现约束

1. 将 K1/K2 的 CHUNK 从 16 改为 32，只适配必要的数据覆盖、分块、归约和计算流程，补齐 Neumann 展开；不改变原算法、MMA 指令、各阶段精度、state 存储及 gate 等其他设置。K1 launch bounds 已按用户授权的对照实验采用 `__launch_bounds__(NumThreads, 4)`。
2. 使用 `fla_kda_ref/naive.py` 对拍，报告 output 和最终 state 的最大绝对误差、相对 L2 误差及 Inf/NaN 情况。不修改参考，不隐藏失败结果，不自行设定误差门槛。
3. 第一版就在 CHUNK 增大可能引起数值问题的位置引入 rescale 参数，初始设为 1，不影响原计算。缩放须配套抵消，保持数学语义；已完成首轮对拍 sweep，当前默认 `rescale=2^-96`、`inverse_rescale=1`，保留显式设置为 1 的未缩放路径。
4. 基于本工作区原有 K1/K2 及其 helper 添加代码，不复制旧候选实现。相关数值讨论参考 `/home/lcpu/60990375/FlashKDA/profile/topic1/`。

验证和按需诊断直接使用对拍文件及其用例，不另建测试套件。
