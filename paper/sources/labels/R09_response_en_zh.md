# R09 source-and-label clarification — response draft

**Use:** This is a response-ready summary for the source and outcome-definition issue. It is not presented as an issued journal decision or an actual reviewer quotation. D assigns final reference numbering and manuscript page/line locations after merging.

## Query 1 — Which source outcome was used, and how was it carried into this analysis?

### English

We now identify the response explicitly as the published 30-min activity value in EasyDesign Table S3. The source study describes plate-level normalization by subtracting the same-plate mean of two no-template negative-control wells and dividing by the mean of two high-activity positive-control wells; the present study inherits those processed values. A record-level audit confirmed exact numerical agreement for all 10,634 Table S3 records. We also verified the linked structure of Table S4: its first 21,268 rows pair the Table S3 30-min value and the separate 20-min-normalized value for those same records, with no mismatches. Only the 30-min field defines the primary response; the auxiliary label and augmentation rows are kept separate (Huang et al., 2024).

**Manuscript action:** The Methods now names the source column, describes the published plate correction, distinguishes the auxiliary 20-min field, and states that the project’s identity partitions change record membership rather than response values.

### 中文核对

现已明确将 EasyDesign Table S3 中公开的30 min活性数值作为响应变量。原研究描述的板内处理是：先减去同板两个无模板阴性对照孔的均值，再除以两个高活性阳性对照孔的均值；本研究直接继承处理后的公开数值。逐记录审计确认，全部10,634条Table S3记录与当前响应值完全一致。我们还核实了Table S4的关联结构：前21,268行将这些记录的Table S3 30 min值与另一个20 min normalized值成对对应，零差异。主响应仅使用30 min列，辅助标签和增强数据均单独区分（Huang et al., 2024）。

**稿件改动：**Methods注明源字段及原研究板内处理，区分20 min辅助字段，并说明本研究身份划分改变的是记录归属而不是标签数值。

## Query 2 — What scale do the labels represent, and how does the EasyDesign prediction wrapper relate to them?

### English

The 30-min values in Table S3 are the supplied activity-label values used for model evaluation. The current extraction copies that column exactly; it does not add an offset, repeat the plate correction or apply the separate time-folding rule. In a distinct code path, the archived EasyDesign inference wrapper adds 4 to regression predictions and clips its deployment output at zero. The common-cohort EasyDesign-adapted comparator used here instead saves raw `model.predict` outputs, so the wrapper operation is not part of either response-label generation or this comparison. SCC therefore evaluates guide-ranking agreement, and MAE/RMSE quantify deviations on the published 30-min activity-label scale (Huang et al., 2024).

The released workbook does not contain the per-well time series or sample-to-plate/replicate map. We therefore report the verified source column and the source paper’s stated control processing, and keep any additional raw-to-label transformation as a targeted author question rather than changing the frozen response.

### 中文核对

Table S3中的30 min数值就是本研究用于评估模型的公开活性标签。当前数据提取逐项复制该列，没有平移数值、重复执行板内校正，也没有应用独立的时间折算规则。另一个独立代码路径中的EasyDesign推理封装会对回归预测加4并将部署输出截断至0；本研究共同队列的EasyDesign-adapted比较器则直接保存`model.predict`原始输出，因此该封装操作既不属于响应标签生成，也没有进入本轮比较。SCC衡量guide排序一致性，MAE/RMSE则表示公开30 min活性标签尺度上的预测偏差（Huang et al., 2024）。

公开工作簿不包含逐孔时间序列或样本到板次/重复的映射。因此，本研究报告已核实的源列和原文所述对照处理；其他原始读数到标签之间的转换作为一个精准作者问题保留，不改动冻结响应值。

## Evidence and integration status

- The Table S3-to-canonical response mapping is verified across all 10,634 records.
- The Table S4 paired-label mapping is verified for all 10,634 source records.
- The source-scale Methods paragraph is ready for D to insert; final figure/table numbers and page/line locations follow the merged manuscript.
- The optional A/source-author questions concern raw trace transformation and plate/replicate mapping; they do not change the verified label-column mapping used by the current nested model comparison.
