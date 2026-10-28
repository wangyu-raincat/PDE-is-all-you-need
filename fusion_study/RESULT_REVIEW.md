# 融合新初值验证：独立结果复核

结论：完整结果与冻结设计一致，**171/178 项通过，维持不合并**。融合修复了一项 AC 小扩散系数偏差，但锁定原弱形式结构也继承了新的粗采样、无噪声 KS 结构错误。不能根据这个结果事后在每个条件选择最好的方法。

本审阅仅读取 JSON、源文件文本和测试日志；没有读取 NPZ 数值场、重新拟合、预测或修改算法。主指标从最终项系数与参考 JSON 独立重算，预测只核对已保存逐轨迹指标的聚合，未重跑原始场误差计算。

## 完整四臂对照

每臂 30 条件，预测列为每条件 3 条最终轨迹相对 L2 均值的汇总，排除首帧。系数向量误差与逐真实项误差分别报告，不能用向量小误差掩盖小系数偏差。

| 方法 | 完整恢复 | 多余项 | 漏项 | 向量误差中位数 | 最坏真实项误差 | 预测中位数 | 最坏条件预测 | 条件预测>10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weak_original | 29/30 | 2 | 0 | 0.052059% | 6.214845% | 1.551522% | 112.326900% | 5 |
| weak_resolved | 30/30 | 0 | 0 | 0.039930% | 37.581952% | 0.404156% | 9.522946% | 0 |
| anchored_best | 29/30 | 2 | 0 | 0.039930% | 37.581952% | 0.404156% | 13.974573% | 1 |
| fused_median | 29/30 | 2 | 0 | 0.039930% | 6.214845% | 0.424269% | 13.974573% | 1 |

120 个配置全部存在，360/360 条最终推进记录成功，没有漏项或推进失败。底层最终初值仅 9 个，采样与噪声条件配对，不能将 360 次推进当作独立实验。10% 预测线是诊断线，不是新增晋级门槛。

## 分族主指标与消融

每格为“完整恢复；该族最坏逐真实项相对误差”。

| 方程族 | weak_original | weak_resolved | anchored_best | fused_median |
|---|---|---|---|---|
| allen_cahn | 10/10 ; 5.833284% | 10/10 ; 37.581952% | 10/10 ; 37.581952% | 10/10 ; 4.178993% |
| low_viscosity_burgers | 10/10 ; 2.408262% | 10/10 ; 2.408262% | 10/10 ; 2.408262% | 10/10 ; 2.408262% |
| ks_unstable | 9/10 ; 6.214845% | 10/10 ; 1.507901% | 9/10 ; 6.214845% | 9/10 ; 6.214845% |

中位数接受 **23** 条，其余 **7** 条只有一个近优变体，保留 anchored_best。池大小计数为 `{2: 11, 3: 12, 1: 7}`。相对 anchored_best，实际 **16** 条系数向量改变，**12** 条条件预测误差上升。因此“接纳融合”不等于最终预测逐条件改善。不存在中位数计算/预测失败后被删除的条件。

AC coarse、1% 噪声、种子 61 中，weak_resolved 和 anchored_best 的扩散项误差为 37.58195%；融合扩散系数为 `9.7715146983e−5`，该项误差降至 **2.28485%**，与这个条件的 weak_original 扩散系数相同。融合整个 AC 家族的最坏误差为 **4.17899%**，来自另一个条件。weak_original 整个 AC 家族原本只有 5.83328% 的最坏误差，融合相对它仅下降 1.65429 个百分点，未达到预声明的 5 个百分点。因此不能只比较较差的 weak_resolved 就宣称明确收益规则通过。

## 七个未通过检查

1. **KS coarse、clean，相对 weak_resolved 的系数向量非劣**：融合 3.650272%，对照 0.921824%，超过 `max(1.25×对照,1%)`。
2. **同条件预测非劣**：融合 13.974573%，对照 3.286828%，超过 `max(1.25×对照,5%)`。
3. **同条件干净项集不得退化**：融合继承原结构锚中的多余项 `u^3`、`u_x`，weak_resolved 恢复正确三项。
4. **相对 weak_resolved 总完整恢复不得下降**：29/30 < 30/30。
5. **相对 weak_resolved 多余项不得增加**：2 > 0。
6. **KS 家族尾部误差保护**：融合 6.21485%，weak_resolved 为 1.50790%，超过 `max(1.25×对照,1%)`。
7. **预声明明确收益**：没有同一家族相对两个直接对照同时达到尾部误差相对下降至少 25%、绝对下降至少 5 个百分点。

前五个检查都由同一个 KS coarse、clean 配置触发；家族尾部保护也受该配置影响。Burgers 与 KS 的尾部误差相对 anchored_best 没有改变，AC 的改善可归因于中位数；这份消融支持局部收益，并不支持整个融合流程无回归。

所有 fused_median 条件的真实项误差仍满足无噪声/0.1% 不超过 10%、1% 不超过 25% 的绝对上限。这个事实与“不合并”并不矛盾：相对更好的对照退化及多余项同样属于预设失败。

## 独立复核内容与证据边界

- 独立构造 3 族×2 采样×5 噪声×4 方法的 120 个配置 ID 集合，检查无缺失、重复或多余配置；各模型训练/开发角色数为 3/2。
- 验证 **17** 个冻结源码/协议摘要、120 个模型文件摘要、准备收据摘要和最终结果引用的 freeze_index 摘要。全部模型依赖与全局冻结清单一致。
- 核对模型未加载最终角色、未用参考方程和干净未来选模的记录，并按源码流程检查：所有模型冻结后才首次生成最终场。这里不是独立运行时追踪认证。
- 从冻结模型核对两种原选择器的近优池/最少项规则、原结构锚、所有同结构成员、每变体去重、中位数及开发接纳。重算 **553** 条已保存候选的两轨迹开发 RMSE 聚合，均吻合。
- 核对每个条件的 weak_original 与 weak_resolved 候选 ID、系数完全相同，差别来自声明的开发预测评分和选择，而不是两个对照分别重训得到不同候选。
- 从项名字典独立重算 120 条完整恢复、多余/漏项、系数向量相对 L2 和逐真实项误差；与记录相同。没有调用项目内主指标函数作为复核捷径。
- 核对全部 360 条逐轨迹记录的初帧排除与带噪首帧标记，重算干净/带噪条件均值及本报告汇总。没有独立重新计算逐时空点误差。
- 根据协议独立计算全部 **178** 个门槛布尔值，与保存结果逐项同名同序一致，171 项通过、7 项未通过。明确收益分母确为每族每臂全部 10 条件，不跨家族拼接两个对照的收益。
- 最终场索引恰好覆盖三族各三个初值、共 9 条 SHA-256 记录；为遵守只审派生证据的范围，本次未读最终 NPZ 文件重算其实际字节摘要。

方法和协议冻结提交为 `807c53357dab868da4621082f7b777f3bd4aae6a`，早于正式训练/开发生成。此次审阅未改变模型、方法、门槛或数据；[冻结代码聚合测试日志](../artifacts/fusion-frozen-code-tests.log)记录 **317 项通过，28.116 秒**。不把本次指标复核条数累加到程序测试数。

另外核对主报告 `REPORT.md` 与 `DECISION.json`：四臂总体/分族数字、171/178、七个失败名称、AC 家族最大值与单条件误差的区分一致；FM 相对 AB 的预测变化确为 12 条上升、4 条下降、14 条不变。主报告所列 WR/FM 最坏单轨迹误差 **12.12277% / 17.70813%** 与逐轨迹记录一致，未与条件平均最大值混淆。DECISION 的结果摘要、冻结摘要、未合并结论及失败列表与原始结果一致。

```text
comparison.json  b422dc07e0f0278e3d6ce2e7737453870eb016b7c18d869a360eb12ef59dea03
freeze_index.json c33e6d013bc25d640782132563e6bf22a8987c6466a6520df0ca40f08faf9108
prepared.json    d69af711a3370bb578e67f7494b9d06f24dbbd99999554aa5cc38e1f89021483
```

这些是既有三个数值 PDE 家族的新初值，不是真实测量或新方程家族。最终数据现在已查看，后续再用它调整融合规则只能称回顾性。当前失败不允许通过事后删项、按最终误差逐条件切换方法或修改明确收益门槛消除。

## 全部 120 条配置

百分数均按上面的指标定义，未省略失败配置。

| 配置 | 完整项集 | 多余项 | 漏项 | 系数向量误差 | 最坏真实项误差 | 条件平均预测误差 |
|---|---|---:|---:|---:|---:|---:|
| allen_cahn--full--clean--weak_original | 是 | 0 | 0 | 0.000001% | 0.000014% | 0.022127% |
| allen_cahn--full--clean--weak_resolved | 是 | 0 | 0 | 0.000001% | 0.000001% | 0.000001% |
| allen_cahn--full--clean--anchored_best | 是 | 0 | 0 | 0.000001% | 0.000001% | 0.000001% |
| allen_cahn--full--clean--fused_median | 是 | 0 | 0 | 0.000001% | 0.000001% | 0.000001% |
| allen_cahn--full--noise0p1_s61--weak_original | 是 | 0 | 0 | 0.013154% | 0.082338% | 0.176030% |
| allen_cahn--full--noise0p1_s61--weak_resolved | 是 | 0 | 0 | 0.013154% | 0.082338% | 0.176266% |
| allen_cahn--full--noise0p1_s61--anchored_best | 是 | 0 | 0 | 0.013154% | 0.082338% | 0.176266% |
| allen_cahn--full--noise0p1_s61--fused_median | 是 | 0 | 0 | 0.012959% | 0.112989% | 0.176213% |
| allen_cahn--full--noise0p1_s97--weak_original | 是 | 0 | 0 | 0.002774% | 0.183430% | 0.183657% |
| allen_cahn--full--noise0p1_s97--weak_resolved | 是 | 0 | 0 | 0.002774% | 0.183430% | 0.183581% |
| allen_cahn--full--noise0p1_s97--anchored_best | 是 | 0 | 0 | 0.002774% | 0.183430% | 0.183581% |
| allen_cahn--full--noise0p1_s97--fused_median | 是 | 0 | 0 | 0.002215% | 0.498030% | 0.182782% |
| allen_cahn--full--noise1_s61--weak_original | 是 | 0 | 0 | 0.135402% | 0.963403% | 1.736968% |
| allen_cahn--full--noise1_s61--weak_resolved | 是 | 0 | 0 | 0.118093% | 11.921595% | 1.710882% |
| allen_cahn--full--noise1_s61--anchored_best | 是 | 0 | 0 | 0.118093% | 11.921595% | 1.710882% |
| allen_cahn--full--noise1_s61--fused_median | 是 | 0 | 0 | 0.135402% | 0.963403% | 1.755001% |
| allen_cahn--full--noise1_s97--weak_original | 是 | 0 | 0 | 0.015607% | 5.833284% | 1.788881% |
| allen_cahn--full--noise1_s97--weak_resolved | 是 | 0 | 0 | 0.015607% | 5.833284% | 1.801016% |
| allen_cahn--full--noise1_s97--anchored_best | 是 | 0 | 0 | 0.015607% | 5.833284% | 1.801016% |
| allen_cahn--full--noise1_s97--fused_median | 是 | 0 | 0 | 0.006395% | 4.178993% | 1.806000% |
| allen_cahn--coarse--clean--weak_original | 是 | 0 | 0 | 0.000123% | 0.027915% | 0.603672% |
| allen_cahn--coarse--clean--weak_resolved | 是 | 0 | 0 | 0.000090% | 0.005974% | 0.000152% |
| allen_cahn--coarse--clean--anchored_best | 是 | 0 | 0 | 0.000090% | 0.005974% | 0.000152% |
| allen_cahn--coarse--clean--fused_median | 是 | 0 | 0 | 0.000090% | 0.005974% | 0.000152% |
| allen_cahn--coarse--noise0p1_s61--weak_original | 是 | 0 | 0 | 0.016014% | 0.491310% | 0.642447% |
| allen_cahn--coarse--noise0p1_s61--weak_resolved | 是 | 0 | 0 | 0.016014% | 0.491310% | 0.229416% |
| allen_cahn--coarse--noise0p1_s61--anchored_best | 是 | 0 | 0 | 0.016014% | 0.491310% | 0.229416% |
| allen_cahn--coarse--noise0p1_s61--fused_median | 是 | 0 | 0 | 0.014783% | 0.219373% | 0.228591% |
| allen_cahn--coarse--noise0p1_s97--weak_original | 是 | 0 | 0 | 0.012135% | 1.566073% | 0.645082% |
| allen_cahn--coarse--noise0p1_s97--weak_resolved | 是 | 0 | 0 | 0.012135% | 1.566073% | 0.241147% |
| allen_cahn--coarse--noise0p1_s97--anchored_best | 是 | 0 | 0 | 0.012135% | 1.566073% | 0.241147% |
| allen_cahn--coarse--noise0p1_s97--fused_median | 是 | 0 | 0 | 0.004354% | 0.236025% | 0.235694% |
| allen_cahn--coarse--noise1_s61--weak_original | 是 | 0 | 0 | 0.173934% | 2.284853% | 2.222852% |
| allen_cahn--coarse--noise1_s61--weak_resolved | 是 | 0 | 0 | 0.149203% | 37.581952% | 2.147129% |
| allen_cahn--coarse--noise1_s61--anchored_best | 是 | 0 | 0 | 0.149203% | 37.581952% | 2.147129% |
| allen_cahn--coarse--noise1_s61--fused_median | 是 | 0 | 0 | 0.173803% | 2.284853% | 2.279487% |
| allen_cahn--coarse--noise1_s97--weak_original | 是 | 0 | 0 | 0.011806% | 1.983170% | 2.292887% |
| allen_cahn--coarse--noise1_s97--weak_resolved | 是 | 0 | 0 | 0.032857% | 4.492526% | 2.309585% |
| allen_cahn--coarse--noise1_s97--anchored_best | 是 | 0 | 0 | 0.032857% | 4.492526% | 2.309585% |
| allen_cahn--coarse--noise1_s97--fused_median | 是 | 0 | 0 | 0.032857% | 1.983170% | 2.348970% |
| low_viscosity_burgers--full--clean--weak_original | 是 | 0 | 0 | 0.000118% | 0.001604% | 0.144819% |
| low_viscosity_burgers--full--clean--weak_resolved | 是 | 0 | 0 | 0.000027% | 0.000435% | 0.000034% |
| low_viscosity_burgers--full--clean--anchored_best | 是 | 0 | 0 | 0.000027% | 0.000435% | 0.000034% |
| low_viscosity_burgers--full--clean--fused_median | 是 | 0 | 0 | 0.000027% | 0.000435% | 0.000034% |
| low_viscosity_burgers--full--noise0p1_s61--weak_original | 是 | 0 | 0 | 0.014609% | 0.014614% | 0.154056% |
| low_viscosity_burgers--full--noise0p1_s61--weak_resolved | 是 | 0 | 0 | 0.014609% | 0.014614% | 0.039984% |
| low_viscosity_burgers--full--noise0p1_s61--anchored_best | 是 | 0 | 0 | 0.014609% | 0.014614% | 0.039984% |
| low_viscosity_burgers--full--noise0p1_s61--fused_median | 是 | 0 | 0 | 0.018186% | 0.018191% | 0.040880% |
| low_viscosity_burgers--full--noise0p1_s97--weak_original | 是 | 0 | 0 | 0.004263% | 0.020209% | 0.152924% |
| low_viscosity_burgers--full--noise0p1_s97--weak_resolved | 是 | 0 | 0 | 0.004263% | 0.020209% | 0.036437% |
| low_viscosity_burgers--full--noise0p1_s97--anchored_best | 是 | 0 | 0 | 0.004263% | 0.020209% | 0.036437% |
| low_viscosity_burgers--full--noise0p1_s97--fused_median | 是 | 0 | 0 | 0.008734% | 0.033194% | 0.037868% |
| low_viscosity_burgers--full--noise1_s61--weak_original | 是 | 0 | 0 | 0.157977% | 0.158012% | 0.432784% |
| low_viscosity_burgers--full--noise1_s61--weak_resolved | 是 | 0 | 0 | 0.157977% | 0.158012% | 0.402538% |
| low_viscosity_burgers--full--noise1_s61--anchored_best | 是 | 0 | 0 | 0.157977% | 0.158012% | 0.402538% |
| low_viscosity_burgers--full--noise1_s61--fused_median | 是 | 0 | 0 | 0.222250% | 0.289521% | 0.418776% |
| low_viscosity_burgers--full--noise1_s97--weak_original | 是 | 0 | 0 | 0.101113% | 0.367410% | 0.416283% |
| low_viscosity_burgers--full--noise1_s97--weak_resolved | 是 | 0 | 0 | 0.101113% | 0.367410% | 0.383560% |
| low_viscosity_burgers--full--noise1_s97--anchored_best | 是 | 0 | 0 | 0.101113% | 0.367410% | 0.383560% |
| low_viscosity_burgers--full--noise1_s97--fused_median | 是 | 0 | 0 | 0.119672% | 0.414598% | 0.390071% |
| low_viscosity_burgers--coarse--clean--weak_original | 是 | 0 | 0 | 0.114360% | 1.596447% | 1.551504% |
| low_viscosity_burgers--coarse--clean--weak_resolved | 是 | 0 | 0 | 0.037286% | 0.453643% | 0.037707% |
| low_viscosity_burgers--coarse--clean--anchored_best | 是 | 0 | 0 | 0.037286% | 0.453643% | 0.037707% |
| low_viscosity_burgers--coarse--clean--fused_median | 是 | 0 | 0 | 0.037286% | 0.453643% | 0.037707% |
| low_viscosity_burgers--coarse--noise0p1_s61--weak_original | 是 | 0 | 0 | 0.118977% | 1.536582% | 1.553817% |
| low_viscosity_burgers--coarse--noise0p1_s61--weak_resolved | 是 | 0 | 0 | 0.045783% | 0.351536% | 0.060654% |
| low_viscosity_burgers--coarse--noise0p1_s61--anchored_best | 是 | 0 | 0 | 0.045783% | 0.351536% | 0.060654% |
| low_viscosity_burgers--coarse--noise0p1_s61--fused_median | 是 | 0 | 0 | 0.045783% | 0.351536% | 0.060654% |
| low_viscosity_burgers--coarse--noise0p1_s97--weak_original | 是 | 0 | 0 | 0.106735% | 1.621512% | 1.551540% |
| low_viscosity_burgers--coarse--noise0p1_s97--weak_resolved | 是 | 0 | 0 | 0.042574% | 0.657383% | 0.064580% |
| low_viscosity_burgers--coarse--noise0p1_s97--anchored_best | 是 | 0 | 0 | 0.042574% | 0.657383% | 0.064580% |
| low_viscosity_burgers--coarse--noise0p1_s97--fused_median | 是 | 0 | 0 | 0.042574% | 0.657383% | 0.064580% |
| low_viscosity_burgers--coarse--noise1_s61--weak_original | 是 | 0 | 0 | 0.019479% | 0.537843% | 1.659207% |
| low_viscosity_burgers--coarse--noise1_s61--weak_resolved | 是 | 0 | 0 | 0.104459% | 0.653224% | 0.481882% |
| low_viscosity_burgers--coarse--noise1_s61--anchored_best | 是 | 0 | 0 | 0.104459% | 0.653224% | 0.481882% |
| low_viscosity_burgers--coarse--noise1_s61--fused_median | 是 | 0 | 0 | 0.104459% | 0.653224% | 0.481882% |
| low_viscosity_burgers--coarse--noise1_s97--weak_original | 是 | 0 | 0 | 0.084639% | 2.408262% | 1.651457% |
| low_viscosity_burgers--coarse--noise1_s97--weak_resolved | 是 | 0 | 0 | 0.084639% | 2.408262% | 0.537947% |
| low_viscosity_burgers--coarse--noise1_s97--anchored_best | 是 | 0 | 0 | 0.084639% | 2.408262% | 0.537947% |
| low_viscosity_burgers--coarse--noise1_s97--fused_median | 是 | 0 | 0 | 0.084639% | 2.408262% | 0.537947% |
| ks_unstable--full--clean--weak_original | 是 | 0 | 0 | 0.000071% | 0.000092% | 0.220648% |
| ks_unstable--full--clean--weak_resolved | 是 | 0 | 0 | 0.000022% | 0.000025% | 0.000120% |
| ks_unstable--full--clean--anchored_best | 是 | 0 | 0 | 0.000022% | 0.000025% | 0.000120% |
| ks_unstable--full--clean--fused_median | 是 | 0 | 0 | 0.000022% | 0.000025% | 0.000120% |
| ks_unstable--full--noise0p1_s61--weak_original | 是 | 0 | 0 | 0.018736% | 0.031135% | 0.463469% |
| ks_unstable--full--noise0p1_s61--weak_resolved | 是 | 0 | 0 | 0.018736% | 0.031135% | 0.405774% |
| ks_unstable--full--noise0p1_s61--anchored_best | 是 | 0 | 0 | 0.018736% | 0.031135% | 0.405774% |
| ks_unstable--full--noise0p1_s61--fused_median | 是 | 0 | 0 | 0.006571% | 0.007276% | 0.429763% |
| ks_unstable--full--noise0p1_s97--weak_original | 是 | 0 | 0 | 0.009011% | 0.013044% | 0.623974% |
| ks_unstable--full--noise0p1_s97--weak_resolved | 是 | 0 | 0 | 0.009011% | 0.013044% | 0.576294% |
| ks_unstable--full--noise0p1_s97--anchored_best | 是 | 0 | 0 | 0.009011% | 0.013044% | 0.576294% |
| ks_unstable--full--noise0p1_s97--fused_median | 是 | 0 | 0 | 0.004715% | 0.007909% | 0.579321% |
| ks_unstable--full--noise1_s61--weak_original | 是 | 0 | 0 | 0.181104% | 0.300524% | 4.053165% |
| ks_unstable--full--noise1_s61--weak_resolved | 是 | 0 | 0 | 0.181104% | 0.300524% | 4.056154% |
| ks_unstable--full--noise1_s61--anchored_best | 是 | 0 | 0 | 0.181104% | 0.300524% | 4.056154% |
| ks_unstable--full--noise1_s61--fused_median | 是 | 0 | 0 | 0.069479% | 0.100866% | 4.167578% |
| ks_unstable--full--noise1_s97--weak_original | 是 | 0 | 0 | 0.346432% | 0.395508% | 5.639112% |
| ks_unstable--full--noise1_s97--weak_resolved | 是 | 0 | 0 | 0.346432% | 0.395508% | 5.629258% |
| ks_unstable--full--noise1_s97--anchored_best | 是 | 0 | 0 | 0.346432% | 0.395508% | 5.629258% |
| ks_unstable--full--noise1_s97--fused_median | 是 | 0 | 0 | 0.346432% | 0.395508% | 5.629258% |
| ks_unstable--coarse--clean--weak_original | 否 | 2 | 0 | 3.650272% | 6.214845% | 85.254386% |
| ks_unstable--coarse--clean--weak_resolved | 是 | 0 | 0 | 0.921824% | 1.473106% | 3.286828% |
| ks_unstable--coarse--clean--anchored_best | 否 | 2 | 0 | 3.650272% | 6.214845% | 13.974573% |
| ks_unstable--coarse--clean--fused_median | 否 | 2 | 0 | 3.650272% | 6.214845% | 13.974573% |
| ks_unstable--coarse--noise0p1_s61--weak_original | 是 | 0 | 0 | 2.469224% | 4.224022% | 93.697609% |
| ks_unstable--coarse--noise0p1_s61--weak_resolved | 是 | 0 | 0 | 0.946716% | 1.507901% | 3.125638% |
| ks_unstable--coarse--noise0p1_s61--anchored_best | 是 | 0 | 0 | 0.946716% | 1.507901% | 3.125638% |
| ks_unstable--coarse--noise0p1_s61--fused_median | 是 | 0 | 0 | 0.946716% | 1.507901% | 3.125638% |
| ks_unstable--coarse--noise0p1_s97--weak_original | 是 | 0 | 0 | 2.636098% | 4.508928% | 91.463525% |
| ks_unstable--coarse--noise0p1_s97--weak_resolved | 是 | 0 | 0 | 0.908686% | 1.453181% | 3.574749% |
| ks_unstable--coarse--noise0p1_s97--anchored_best | 是 | 0 | 0 | 0.908686% | 1.453181% | 3.574749% |
| ks_unstable--coarse--noise0p1_s97--fused_median | 是 | 0 | 0 | 0.908686% | 1.453181% | 3.574749% |
| ks_unstable--coarse--noise1_s61--weak_original | 是 | 0 | 0 | 0.552251% | 0.945751% | 112.326900% |
| ks_unstable--coarse--noise1_s61--weak_resolved | 是 | 0 | 0 | 0.552251% | 0.945751% | 6.693899% |
| ks_unstable--coarse--noise1_s61--anchored_best | 是 | 0 | 0 | 0.552251% | 0.945751% | 6.693899% |
| ks_unstable--coarse--noise1_s61--fused_median | 是 | 0 | 0 | 0.552251% | 0.945751% | 6.693899% |
| ks_unstable--coarse--noise1_s97--weak_original | 是 | 0 | 0 | 2.231244% | 3.807639% | 91.284293% |
| ks_unstable--coarse--noise1_s97--weak_resolved | 是 | 0 | 0 | 0.785473% | 1.266970% | 9.522946% |
| ks_unstable--coarse--noise1_s97--anchored_best | 是 | 0 | 0 | 0.785473% | 1.266970% | 9.522946% |
| ks_unstable--coarse--noise1_s97--fused_median | 是 | 0 | 0 | 0.785473% | 1.266970% | 9.522946% |
