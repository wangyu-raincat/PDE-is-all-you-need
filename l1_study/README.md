# 独立 L1 方法评估分支

目标是借鉴 changsongjin/pde-l1-analytic 的方法，实证比较方程项与系数恢复；生产目录 /workspace/pde-discovery 保持原样。原项目没有 Git 历史，本轮在隔离副本建立 baseline/frozen-generic 与 experiment/l1-analytic-assessment 分支。基线提交 b8f24a456ce63253994d05e446528c54ac88d32e；生产代码没有改动。协议为 protocol.json，算法固定常数见 engine.PROTOCOL。

## 参考修正与方法

固定参考提交 271c6ab556c0788cad5d60e86537c8b0d6dab9e0。redo_two.py 第84行应为 u_xxx = u*(3*xi/s**2 - xi**3/s**3)，其中 s=sigma0**2+2*nu*t、xi=x-x0-c*t。最小补丁、来源哈希、审查记录在 reference_review/；不会执行整份第三方脚本。可运行 python -m l1_study.gaussian_check 重现独立复步长检查。仓库没有明确再分发许可，交付不复制完整作者源码。

参考库只有一次 adaptive L1；本分支独立实现固定5轮 IRL1。共同9项库、同一alpha路径、训练RMS归一化和支持内OLS去偏。权重由训练岭回归初始化，alpha=1e-6，w=clip(1/(abs(beta)+1e-3),1e-2,1e3)。每个lambda重新从同一训练pilot开始。没有跨lambda比较惩罚目标，没有路径频次投票，没有按参考方程筛项。

## 预定对照

- 旧 Burgers/KdV/KS：保留原60/20/20时间段与积分特征，plain/adaptive/IRL1，共9模型。所有最终段以前均已查看，明确标为回顾性比较；不会把单轨迹切片伪称多轨迹。
- 新解析轨迹：对流扩散、线性二至四阶方程、Cole–Hopf Burgers，共三族。每族3训练初值、2开发初值、2最终初值；方程参数固定。所有初值都含4个模式，不故意用单模态制造联合方法优势。
- 3种回归输入 × 1/3训练轨迹 × 3种L1，共54模型。三个输入为原积分、估计导数点态、解析导数点态。点态精确对照同时消除数值求导和谱投影差异，不能将全部差异单独归因求导。精确导数是解析信息上限，不是从真实观测可获得的输入。
- 同一输入类型中单/多轨迹采用同一总回归行预算，固定linspace行索引、每条轨迹等权；共同测试新初值。不同输入类型的行预算可能不同。联合方法增加了初值多样性，结果不能解释为普适算法优势。

全部模型只由训练和开发决定。开发选模为各轨迹MSE等权平均再开方，最小RMSE的1.05倍加1e-12内选最少项，再按误差/固定ID决胜。63模型全部冻结且源码、协议、数据和模型哈希校验通过，才生成新最终初值场、读取旧最终角色并计分。最终不重选、不重拟合、不调整阈值。参考函数与发现接口分离；数据生成器当然知道预定方程，因此不称完全未知物理盲发现。

主要指标是精确项集合匹配、漏项/多余项数、统一候选坐标下系数向量相对L2；自由推进误差为辅助。所有额外项照计，零阈值只在训练阶段预先固定。无实际测量、噪声鲁棒性或任意PDE保证。

## 合入规则

事先指定 IRL1 为唯一候选；adaptive 是对照，不按最终成绩另选赢家。旧三组至少精确恢复2/3、总多余项不超过2；所有非oracle配置不得增加漏项/推进失败、不得把plain成功项集变成失败。每例系数误差不超过 max(1.10×plain,1e-4)，预测误差不超过 max(1.25×plain,1e-3)。新解析非oracle配置至少一例结构严格改善。旧128项及新回归测试必须全部通过。此为保守工程门槛，不是统计显著性检验；不满足就不合入，原目录继续为活动基线。

## 复现

在此隔离项目根目录安装 requirements-repro.txt。可直接阅读 results/first_run/comparison.json 和本轮 REPORT.md；复现首次结果不产生新的未见验证。

```bash
python -m l1_study.gaussian_check
python -m unittest discover -s tests -v
python -m pde_discovery generic-benchmark --phase download
python -m pde_discovery generic-benchmark --phase prepare
python -m l1_study.run prepare
python -m l1_study.run freeze --output l1_study/results/reproduction
python -m l1_study.run evaluate --output l1_study/results/reproduction
```

交付包不含第三方原始MAT/观测NPZ；上述固定版本下载器和准备命令恢复它们。若 l1_study/data/prepared.json 已存在且代码未改，跳过 study prepare，不覆盖原准备数据。freeze/evaluate 拒绝覆写已冻结/已评分输出，使用新的目录复现。源代码或协议变动必须作为下一轮实验，并保留旧记录。当前原型主应用启动方式不变；新方法没有偷偷替换默认引擎。
