# 特征工程详解

> 本文档列出项目中所有工程特征的计算方法和业务含义。
> 原始字段含义见 [data_dictionary.md](data_dictionary.md)。

---

## 特征总览

| 特征组 | 数量 | 数据来源 |
|--------|------|----------|
| 基础衍生特征 | 14 | 主表字段计算 |
| 征信机构聚合 | 22 | bureau.csv |
| 历史申请聚合 | 28 | previous_application.csv |
| 信用卡聚合 | 24 | credit_card_balance.csv |
| 还款记录聚合 | 18 | installments_payments.csv |
| POS/现金贷款聚合 | 12 | POS_CASH_balance.csv |
| **合计** | **118** | |

特征命名规则：`{来源}_{原始字段}_{聚合方式}`

- `BUREAU_` = 征信机构
- `PREV_` = 历史申请
- `CC_` = 信用卡
- `IP_` = 还款记录
- `POS_` = POS/现金贷款

---

## A. 基础衍生特征（14 个）

从主表字段直接计算，不需要关联其他表。

### 债务负担类

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| CREDIT_INCOME_RATIO | AMT_CREDIT / (AMT_INCOME_TOTAL + 1) | 贷款金额占年收入的比例。越高说明负债越重，风险越高。例：值为 5 表示贷款额是年收入的 5 倍。 |
| ANNUITY_INCOME_RATIO | AMT_ANNUITY / (AMT_INCOME_TOTAL + 1) | 年还款额占年收入的比例。衡量月供压力。值 > 0.5 表示超过一半收入用于还贷。 |

### 年龄与工龄类

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| AGE_YEARS | -DAYS_BIRTH / 365 | 申请时年龄（岁）。DAYS_BIRTH 为负值，取反再除以 365。 |
| AGE_GROUP | pd.cut(AGE_YEARS, [0,25,35,45,55,100]) → 0-4 | 年龄分段：0=<25 岁，1=25-35，2=35-45，3=45-55，4=55+。年轻和年老申请人风险特征不同。 |
| EMPLOYMENT_YEARS | -DAYS_EMPLOYED / 365 | 当前工作年限。工作越稳定，违约风险越低。 |
| INCOME_PER_EMPLOYMENT_YEAR | AMT_INCOME_TOTAL / (EMPLOYMENT_YEARS + 1) | 每年工龄对应的收入。反映职业成长速度。值高说明要么高薪要么刚入职。 |

### 家庭类

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| CHILDREN_RATIO | CNT_CHILDREN / (CNT_FAM_MEMBERS + 1) | 子女占家庭成员比例。值高说明家庭负担重。 |

### 车辆类

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| OWN_CAR_AGE_MISSING | OWN_CAR_AGE.isnull() → 0/1 | 是否没车。缺失值本身就是信号：没车的人可能经济状况不同。 |
| CAR_AGE_FILLED | OWN_CAR_AGE.fillna(0) | 车龄（没车填 0）。 |

### 文件与外部评分类

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| DOCUMENT_COUNT | sum(FLAG_DOCUMENT_2 ~ FLAG_DOCUMENT_21) | 提交文件数量。提交太多或太少都可能异常。 |
| EXT_SOURCE_MEAN | mean(EXT_SOURCE_1, 2, 3) | 三个外部征信评分的均值。**预测力最强的特征**（IV=0.63）。外部评分来自第三方征信机构，综合评估信用风险。 |
| EXT_SOURCE_STD | std(EXT_SOURCE_1, 2, 3) | 外部评分的标准差。值高说明不同征信机构对同一人的评价分歧大，存在信息不对称。 |
| EXT_SOURCE_MIN | min(EXT_SOURCE_1, 2, 3) | 最低外部评分。"木桶效应"——最差的那个评分最能反映风险。 |
| EXT_SOURCE_MAX | max(EXT_SOURCE_1, 2, 3) | 最高外部评分。 |

---

## B. 征信机构聚合特征（22 个）

数据来源：`bureau.csv`，按 `SK_ID_CURR`（客户 ID）分组聚合。

### 信贷历史时间特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| BUREAU_DAYS_CREDIT_MIN | min(DAYS_CREDIT) | 最近一次征信记录。值越大（越接近 0）说明最近有新信贷活动。 |
| BUREAU_DAYS_CREDIT_MAX | max(DAYS_CREDIT) | 最早一次征信记录。值越小说明信贷历史越长。 |
| BUREAU_DAYS_CREDIT_MEAN | mean(DAYS_CREDIT) | 平均征信记录时间。 |

### 逾期特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| BUREAU_CREDIT_DAY_OVERDUE_MAX | max(CREDIT_DAY_OVERDUE) | 历史最大逾期天数。有逾期记录是强风险信号。 |
| BUREAU_CREDIT_DAY_OVERDUE_MEAN | mean(CREDIT_DAY_OVERDUE) | 平均逾期天数。 |
| BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX | max(AMT_CREDIT_MAX_OVERDUE) | 历史最大逾期金额。 |
| BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN | mean(AMT_CREDIT_MAX_OVERDUE) | 平均最大逾期金额。 |
| BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM | sum(AMT_CREDIT_SUM_OVERDUE) | 所有征信信贷的当前逾期总额。 |
| BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN | mean(AMT_CREDIT_SUM_OVERDUE) | 平均每笔逾期金额。 |
| BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX | max(AMT_CREDIT_SUM_OVERDUE) | 单笔最大逾期金额。 |

### 信贷额度与债务特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| BUREAU_DAYS_CREDIT_ENDDATE_MIN | min(DAYS_CREDIT_ENDDATE) | 最近到期的征信信贷。 |
| BUREAU_DAYS_CREDIT_ENDDATE_MAX | max(DAYS_CREDIT_ENDDATE) | 最远到期的征信信贷。 |
| BUREAU_DAYS_CREDIT_ENDDATE_MEAN | mean(DAYS_CREDIT_ENDDATE) | 平均剩余到期时间。 |
| BUREAU_AMT_CREDIT_SUM_SUM | sum(AMT_CREDIT_SUM) | 征信信贷总额。反映申请人在其他机构的总授信。 |
| BUREAU_AMT_CREDIT_SUM_MEAN | mean(AMT_CREDIT_SUM) | 平均每笔信贷金额。 |
| BUREAU_AMT_CREDIT_SUM_MAX | max(AMT_CREDIT_SUM) | 单笔最大信贷金额。 |
| BUREAU_AMT_CREDIT_SUM_DEBT_SUM | sum(AMT_CREDIT_SUM_DEBT) | 征信未偿还债务总额。值高说明多头借贷。 |
| BUREAU_AMT_CREDIT_SUM_DEBT_MEAN | mean(AMT_CREDIT_SUM_DEBT) | 平均每笔未偿还债务。 |
| BUREAU_AMT_CREDIT_SUM_DEBT_MAX | max(AMT_CREDIT_SUM_DEBT) | 单笔最大未偿还债务。 |

### 计数与比率特征

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| BUREAU_LOAN_COUNT | count(征信记录) | 征信信贷总数。多头借贷风险。 |
| BUREAU_ACTIVE_LOAN_COUNT | count(CREDIT_ACTIVE == 'Active') | 当前活跃的征信信贷数。 |
| BUREAU_DEBT_CREDIT_RATIO | DEBT_SUM / (CREDIT_SUM + 1) | 征信信贷使用率。值接近 1 说明额度几乎用完，风险高。 |

---

## C. 历史申请聚合特征（28 个）

数据来源：`previous_application.csv`，按 `SK_ID_CURR` 分组聚合。

### 申请金额特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| PREV_AMT_ANNUITY_MIN/MAX/MEAN | min/max/mean(AMT_ANNUITY) | 历史申请的年还款额范围。 |
| PREV_AMT_APPLICATION_MIN/MAX/MEAN | min/max/mean(AMT_APPLICATION) | 历史申请金额范围。申请金额逐次增加可能说明资金紧张。 |
| PREV_AMT_CREDIT_MIN/MAX/MEAN | min/max/mean(AMT_CREDIT) | 历史批准金额范围。 |
| PREV_AMT_DOWN_PAYMENT_MIN/MAX/MEAN | min/max/mean(AMT_DOWN_PAYMENT) | 历史首付金额。首付高说明资金充裕。 |
| PREV_RATE_DOWN_PAYMENT_MIN/MAX/MEAN | min/max/mean(RATE_DOWN_PAYMENT) | 历史首付比例。 |

### 时间特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| PREV_DAYS_DECISION_MIN | min(DAYS_DECISION) | 最近一次历史申请。值越大（越接近 0）说明最近有申请。 |
| PREV_DAYS_DECISION_MAX | max(DAYS_DECISION) | 最早一次历史申请。 |
| PREV_DAYS_DECISION_MEAN | mean(DAYS_DECISION) | 平均申请时间。 |
| PREV_HOUR_APPR_PROCESS_START_MIN/MAX/MEAN | min/max/mean(HOUR) | 申请时段分布。 |

### 还款期限特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| PREV_CNT_PAYMENT_MEAN | mean(CNT_PAYMENT) | 平均贷款期限。期限变长可能说明还款能力下降。 |
| PREV_CNT_PAYMENT_SUM | sum(CNT_PAYMENT) | 历史贷款总期数。 |

### 审批结果特征

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| PREV_APP_COUNT | count(历史申请) | 历史申请总次数。频繁申请是风险信号。 |
| PREV_APPROVED_COUNT | count(STATUS == 'Approved') | 被批准的次数。 |
| PREV_REFUSED_COUNT | count(STATUS == 'Refused') | 被拒绝的次数。多次被拒说明信用有问题。 |
| PREV_APPROVAL_RATE | APPROVED / (APP_COUNT + 1) | 历史通过率。 |
| PREV_REFUSED_RATE | REFUSED / (APP_COUNT + 1) | 历史拒绝率。值高说明信用差。 |

---

## D. 信用卡聚合特征（24 个）

数据来源：`credit_card_balance.csv`（月度快照），按 `SK_ID_CURR` 分组聚合。

### 余额特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| CC_AMT_BALANCE_MEAN | mean(AMT_BALANCE) | 平均月度信用卡余额。 |
| CC_AMT_BALANCE_MAX | max(AMT_BALANCE) | 最高月度余额。 |
| CC_AMT_BALANCE_MIN | min(AMT_BALANCE) | 最低月度余额。 |
| CC_AMT_BALANCE_STD | std(AMT_BALANCE) | 余额波动性。波动大说明消费不稳定。 |

### 额度特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN | mean(LIMIT) | 平均信用卡额度。额度高说明银行信任度高。 |
| CC_AMT_CREDIT_LIMIT_ACTUAL_MAX | max(LIMIT) | 最高信用卡额度。 |

### 取现特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| CC_AMT_DRAWINGS_CURRENT_MEAN | mean(DRAWINGS) | 平均月取现金额。取现多说明现金流紧张。 |
| CC_AMT_DRAWINGS_CURRENT_MAX | max(DRAWINGS) | 单月最大取现金额。 |
| CC_CNT_DRAWINGS_CURRENT_MEAN | mean(CNT_DRAWINGS) | 平均月取现次数。 |
| CC_CNT_DRAWINGS_CURRENT_MAX | max(CNT_DRAWINGS) | 单月最大取现次数。 |

### 还款特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| CC_AMT_PAYMENT_CURRENT_MEAN | mean(PAYMENT) | 平均月还款金额。 |
| CC_AMT_PAYMENT_CURRENT_MAX | max(PAYMENT) | 单月最大还款金额。 |
| CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN | mean(PAYMENT_TOTAL) | 平均总还款金额（含分期）。 |

### 应收特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| CC_AMT_TOTAL_RECEIVABLE_MEAN | mean(RECEIVABLE) | 平均总应收金额。 |
| CC_AMT_TOTAL_RECEIVABLE_MAX | max(RECEIVABLE) | 最大总应收金额。 |

### 逾期特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| CC_SK_DPD_MEAN | mean(SK_DPD) | 平均逾期天数（含宽限期）。 |
| CC_SK_DPD_MAX | max(SK_DPD) | 最大逾期天数。 |
| CC_SK_DPD_DEF_MEAN | mean(SK_DPD_DEF) | 平均逾期天数（不含宽限期）。 |
| CC_SK_DPD_DEF_MAX | max(SK_DPD_DEF) | 最大逾期天数（不含宽限期）。 |

### 比率特征

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| CC_UTILIZATION_RATIO | BALANCE_MEAN / (LIMIT_MEAN + 1) | **额度使用率**。>80% 是高风险信号，说明信用卡几乎刷爆。 |
| CC_PAYMENT_RATIO | PAYMENT_MEAN / (BALANCE_MEAN + 1) | **还款比率**。<100% 说明未全额还款，有利息支出。 |
| CC_MONTHS_COUNT | count(月度记录) | 信用卡记录月数。 |
| CC_DPD_RATIO | count(DPD > 0) / TOTAL | **逾期月份占比**。反映逾期频率。 |
| CC_ACTIVE_RATIO | count(Active) / TOTAL | 活跃账户占比。多账户活跃 → 多头借贷。 |

---

## E. 还款记录聚合特征（18 个）

数据来源：`installments_payments.csv`，按 `SK_ID_CURR` 分组聚合。

### 中间衍生列（逐行计算）

| 列名 | 计算公式 | 含义 |
|------|----------|------|
| PAYMENT_DELAY | DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT | 还款延迟天数。正值=晚还，负值=早还。 |
| PAYMENT_DIFF | AMT_PAYMENT - AMT_INSTALMENT | 还款差额。负值=少还。 |
| PAYMENT_RATIO | AMT_PAYMENT / (AMT_INSTALMENT + 1) | 还款完成度。>1=多还，<1=少还。 |

### 延迟特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| IP_PAYMENT_DELAY_MEAN | mean(DELAY) | 平均还款延迟天数。正值说明习惯性晚还。 |
| IP_PAYMENT_DELAY_MAX | max(DELAY) | 最长延迟天数。 |
| IP_PAYMENT_DELAY_MIN | min(DELAY) | 最早提前还款天数。 |
| IP_PAYMENT_DELAY_STD | std(DELAY) | 还款时间波动性。 |

### 差额特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| IP_PAYMENT_DIFF_MEAN | mean(DIFF) | 平均还款差额。负值说明经常少还。 |
| IP_PAYMENT_DIFF_MAX | max(DIFF) | 最大多还金额。 |
| IP_PAYMENT_DIFF_MIN | min(DIFF) | 最大少还金额。 |

### 完成度特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| IP_PAYMENT_RATIO_MEAN | mean(RATIO) | 平均还款完成度。 |
| IP_PAYMENT_RATIO_MIN | min(RATIO) | 最差还款完成度。 |

### 金额特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| IP_NUM_INSTALMENT_VERSION_MAX | max(VERSION) | 还款计划最大版本。版本变化说明经历过重组/再融资。 |
| IP_AMT_INSTALMENT_SUM | sum(INSTALMENT) | 应还总额。 |
| IP_AMT_INSTALMENT_MEAN | mean(INSTALMENT) | 平均每期应还。 |
| IP_AMT_PAYMENT_SUM | sum(PAYMENT) | 实还总额。 |
| IP_AMT_PAYMENT_MEAN | mean(PAYMENT) | 平均每期实还。 |

### 计数特征

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| IP_LATE_COUNT | count(DELAY > 0) | 逾期次数。 |
| IP_LATE_RATIO | LATE_COUNT / (TOTAL + 1) | **逾期比例**。核心风险指标。 |
| IP_UNDERPAY_COUNT | count(DIFF < 0) | 少还次数。资金不足信号。 |
| IP_TOTAL_COUNT | count(记录) | 总还款期数。 |

---

## F. POS/现金贷款聚合特征（12 个）

数据来源：`POS_CASH_balance.csv`（月度快照），按 `SK_ID_CURR` 分组聚合。

### 逾期特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| POS_SK_DPD_MEAN | mean(SK_DPD) | 平均逾期天数。 |
| POS_SK_DPD_MAX | max(SK_DPD) | 最大逾期天数。 |
| POS_SK_DPD_DEF_MEAN | mean(SK_DPD_DEF) | 平均逾期天数（不含宽限期）。 |
| POS_SK_DPD_DEF_MAX | max(SK_DPD_DEF) | 最大逾期天数（不含宽限期）。 |

### 分期特征

| 特征 | 聚合方式 | 业务含义 |
|------|----------|----------|
| POS_CNT_INSTALMENT_MEAN | mean(CNT_INSTALMENT) | 平均总分期数。 |
| POS_CNT_INSTALMENT_MAX | max(CNT_INSTALMENT) | 最大总分期数。 |
| POS_CNT_INSTALMENT_FUTURE_MEAN | mean(FUTURE) | 平均剩余分期数。 |
| POS_CNT_INSTALMENT_FUTURE_MAX | max(FUTURE) | 最大剩余分期数。 |

### 比率特征

| 特征 | 计算公式 | 业务含义 |
|------|----------|----------|
| POS_DPD_RATIO | count(DPD > 0) / TOTAL | 逾期月份占比。 |
| POS_INSTALLMENT_PROGRESS | mean(INSTALMENT - FUTURE) | 平均还款进度。值越大说明还得越多。 |
| POS_MONTHS_COUNT | count(月度记录) | POS/现金贷款记录月数。 |
| POS_ACTIVE_RATIO | count(Active) / TOTAL | 活跃账户占比。多头借贷信号。 |

---

## 数据处理补充说明

### 编码方式

| 类别数量 | 编码方法 | 说明 |
|----------|----------|------|
| 2 种 | Label Encoding → 0/1 | 如性别 M/F |
| 3-10 种 | One-Hot Encoding | 如教育程度，生成虚拟变量列 |
| 11+ 种 | Frequency Encoding | 用该类别出现的比例替代（默认） |

### 缺失值处理

1. 缺失率 > 80% 的列直接删除
2. 数值列：按 `NAME_INCOME_TYPE` 分组中位数填充 → 全局中位数兜底 → 0 填充残留
3. 类别列：众数填充

### 异常值处理

对以下列使用 IQR 方法截尾（Q1 - 3×IQR ~ Q3 + 3×IQR）：
`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `DAYS_EMPLOYED`, `DAYS_BIRTH`, `OWN_CAR_AGE`

### 高相关特征删除

计算 Pearson 相关系数矩阵，相关系数 > 0.95 的特征对中删除后出现的那个。从 277 个特征删到 222 个。
