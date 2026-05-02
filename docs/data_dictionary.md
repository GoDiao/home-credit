# 数据字典

> 本文档列出 Home Credit 数据集所有表的全部字段含义。
> 数据来源：[Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

---

## 表关系总览

```
application_train/test (主表，每行 = 一个贷款申请)
    │
    ├── bureau (该申请人在其他机构的历史信贷)
    │     └── bureau_balance (每月还款状态)
    │
    └── previous_application (该申请人在 Home Credit 的历史申请)
            ├── credit_card_balance (信用卡月度账单)
            ├── installments_payments (每期还款记录)
            └── POS_CASH_balance (POS/现金贷款月度状态)
```

| 表名 | 行数 | 列数 | 关联键 |
|------|------|------|--------|
| application_train | 307,511 | 122 | SK_ID_CURR (主键) |
| application_test | 48,744 | 121 | SK_ID_CURR (主键) |
| bureau | 1,716,428 | 17 | SK_ID_CURR → application; SK_ID_BUREAU → bureau_balance |
| bureau_balance | 27,299,925 | 3 | SK_ID_BUREAU → bureau |
| credit_card_balance | 3,840,312 | 23 | SK_ID_CURR → application; SK_ID_PREV → previous_application |
| installments_payments | 13,605,401 | 8 | SK_ID_CURR → application; SK_ID_PREV → previous_application |
| POS_CASH_balance | 10,001,358 | 8 | SK_ID_CURR → application; SK_ID_PREV → previous_application |
| previous_application | 1,670,214 | 37 | SK_ID_CURR → application; SK_ID_PREV (主键) |

---

## 1. application_train / application_test（主表）

每行代表一个贷款申请。训练集有 TARGET 列（标签），测试集没有。

| 字段 | 类型 | 含义 |
|------|------|------|
| SK_ID_CURR | int | 贷款申请 ID |
| TARGET | int | **(仅训练集)** 1=违约，0=正常 |
| NAME_CONTRACT_TYPE | str | 贷款类型：Cash loans / Revolving loans |
| CODE_GENDER | str | 性别：M / F / XNA |
| FLAG_OWN_CAR | str | 是否有车：Y / N |
| FLAG_OWN_REALTY | str | 是否有房产：Y / N |
| CNT_CHILDREN | int | 子女数量 |
| AMT_INCOME_TOTAL | float | 年收入 |
| AMT_CREDIT | float | 贷款金额 |
| AMT_ANNUITY | float | 年还款额（月供×12） |
| AMT_GOODS_PRICE | float | 消费贷款的商品价格 |
| NAME_TYPE_SUITE | str | 申请时陪同人员 |
| NAME_INCOME_TYPE | str | 收入类型：Working / Commercial associate / Pensioner 等 |
| NAME_EDUCATION_TYPE | str | 学历：Secondary / Higher / Academic degree 等 |
| NAME_FAMILY_STATUS | str | 婚姻状况：Married / Single / Separated 等 |
| NAME_HOUSING_TYPE | str | 住房情况：House / apartment / With parents 等 |
| REGION_POPULATION_RELATIVE | float | 居住地人口密度（归一化） |
| DAYS_BIRTH | int | 申请时年龄（天数，负值） |
| DAYS_EMPLOYED | int | 当前工作开始距今天数（负值；365243=退休） |
| DAYS_REGISTRATION | int | 最近一次更改注册距今天数（负值） |
| DAYS_ID_PUBLISH | int | 最近一次更换身份证距今天数（负值） |
| OWN_CAR_AGE | float | 车龄（年） |
| FLAG_MOBIL | int | 是否提供手机：1=是 / 0=否 |
| FLAG_EMP_PHONE | int | 是否提供工作电话 |
| FLAG_WORK_PHONE | int | 是否提供家庭电话 |
| FLAG_CONT_MOBILE | int | 手机是否可接通 |
| FLAG_PHONE | int | 是否提供座机 |
| FLAG_EMAIL | int | 是否提供邮箱 |
| OCCUPATION_TYPE | str | 职业类型：Laborers / Sales staff / Core staff 等 |
| CNT_FAM_MEMBERS | float | 家庭成员数 |
| REGION_RATING_CLIENT | int | 地区评级（1-3） |
| REGION_RATING_CLIENT_W_CITY | int | 地区评级（含城市，1-3） |
| WEEKDAY_APPR_PROCESS_START | str | 申请星期几 |
| HOUR_APPR_PROCESS_START | int | 申请时段（小时） |
| REG_REGION_NOT_LIVE_REGION | int | 户籍地 ≠ 居住地（省级） |
| REG_REGION_NOT_WORK_REGION | int | 户籍地 ≠ 工作地（省级） |
| LIVE_REGION_NOT_WORK_REGION | int | 居住地 ≠ 工作地（省级） |
| REG_CITY_NOT_LIVE_CITY | int | 户籍地 ≠ 居住地（市级） |
| REG_CITY_NOT_WORK_CITY | int | 户籍地 ≠ 工作地（市级） |
| LIVE_CITY_NOT_WORK_CITY | int | 居住地 ≠ 工作地（市级） |
| ORGANIZATION_TYPE | str | 工作单位类型（58 种） |
| EXT_SOURCE_1 | float | 外部征信评分 1（归一化，越高越安全） |
| EXT_SOURCE_2 | float | 外部征信评分 2 |
| EXT_SOURCE_3 | float | 外部征信评分 3 |
| APARTMENTS_AVG | float | 住房面积均值（归一化） |
| BASEMENTAREA_AVG | float | 地下室面积均值 |
| YEARS_BEGINEXPLUATATION_AVG | float | 建筑使用年数均值 |
| YEARS_BUILD_AVG | float | 建筑建造年数均值 |
| COMMONAREA_AVG | float | 公共区域面积均值 |
| ELEVATORS_AVG | float | 电梯数量均值 |
| ENTRANCES_AVG | float | 入口数量均值 |
| FLOORSMAX_AVG | float | 最大楼层数均值 |
| FLOORSMIN_AVG | float | 最小楼层数均值 |
| LANDAREA_AVG | float | 占地面积均值 |
| LIVINGAPARTMENTS_AVG | float | 居住面积均值 |
| LIVINGAREA_AVG | float | 生活面积均值 |
| NONLIVINGAPARTMENTS_AVG | float | 非居住面积均值 |
| NONLIVINGAREA_AVG | float | 非生活面积均值 |
| APARTMENTS_MODE ~ NONLIVINGAREA_MODE | float | 同上，众数（14 个字段） |
| APARTMENTS_MEDI ~ NONLIVINGAREA_MEDI | float | 同上，中位数（14 个字段） |
| FONDKAPREMONT_MODE | str | 物业维修基金类型 |
| HOUSETYPE_MODE | str | 房屋类型 |
| TOTALAREA_MODE | float | 总面积（归一化） |
| WALLSMATERIAL_MODE | str | 墙体材料 |
| EMERGENCYSTATE_MODE | str | 是否有应急通道 |
| OBS_30_CNT_SOCIAL_CIRCLE | float | 社交圈中可观察到的 30 天逾期人数 |
| DEF_30_CNT_SOCIAL_CIRCLE | float | 社交圈中 30 天逾期违约人数 |
| OBS_60_CNT_SOCIAL_CIRCLE | float | 社交圈中可观察到的 60 天逾期人数 |
| DEF_60_CNT_SOCIAL_CIRCLE | float | 社交圈中 60 天逾期违约人数 |
| DAYS_LAST_PHONE_CHANGE | int | 最近换手机号距今天数 |
| FLAG_DOCUMENT_2 ~ FLAG_DOCUMENT_21 | int | 是否提供第 N 份文件（20 个 0/1 标志） |
| AMT_REQ_CREDIT_BUREAU_HOUR | float | 申请前 1 小时内征信查询次数 |
| AMT_REQ_CREDIT_BUREAU_DAY | float | 申请前 1 天内征信查询次数 |
| AMT_REQ_CREDIT_BUREAU_WEEK | float | 申请前 1 周内征信查询次数 |
| AMT_REQ_CREDIT_BUREAU_MON | float | 申请前 1 月内征信查询次数 |
| AMT_REQ_CREDIT_BUREAU_QRT | float | 申请前 1 季度内征信查询次数 |
| AMT_REQ_CREDIT_BUREAU_YEAR | float | 申请前 1 年内征信查询次数 |

---

## 2. bureau（征信机构历史记录）

每行代表申请人在其他金融机构的一笔信贷记录。

| 字段 | 类型 | 含义 |
|------|------|------|
| SK_ID_CURR | int | 关联到主表的客户 ID |
| SK_ID_BUREAU | int | 征信记录 ID（关联到 bureau_balance） |
| CREDIT_ACTIVE | str | 信贷状态：Active / Closed / Sold / Bad debt |
| CREDIT_CURRENCY | str | 币种 |
| DAYS_CREDIT | int | 征信记录距今天数（负值） |
| CREDIT_DAY_OVERDUE | int | 当前逾期天数 |
| DAYS_CREDIT_ENDDATE | float | 信贷到期剩余天数 |
| DAYS_ENDDATE_FACT | float | 信贷实际结束距今天数（仅已关闭） |
| AMT_CREDIT_MAX_OVERDUE | float | 历史最大逾期金额 |
| CNT_CREDIT_PROLONG | int | 信贷延期次数 |
| AMT_CREDIT_SUM | float | 当前信贷总额 |
| AMT_CREDIT_SUM_DEBT | float | 当前未偿还债务 |
| AMT_CREDIT_SUM_LIMIT | float | 信用卡额度 |
| AMT_CREDIT_SUM_OVERDUE | float | 当前逾期金额 |
| CREDIT_TYPE | str | 信贷类型：Consumer / Car / Mortgage / Credit card 等 |
| DAYS_CREDIT_UPDATE | int | 最近一次征信信息更新距今天数 |
| AMT_ANNUITY | float | 征信信贷年还款额 |

---

## 3. bureau_balance（征信月度状态）

每行代表某笔征信信贷的某月还款状态。

| 字段 | 类型 | 含义 |
|------|------|------|
| SK_ID_BUREAU | int | 关联到 bureau 表 |
| MONTHS_BALANCE | int | 月份（负值，-1=最近一个月） |
| STATUS | str | 还款状态：C=已关闭，X=未知，0=无逾期，1=逾期1-30天，2=逾期31-60天，...5=逾期120+天 |

---

## 4. credit_card_balance（信用卡月度账单）

每行代表某张信用卡的某月账单快照。

| 字段 | 类型 | 含义 |
|------|------|------|
| SK_ID_PREV | int | 关联到 previous_application |
| SK_ID_CURR | int | 关联到主表 |
| MONTHS_BALANCE | int | 月份（负值） |
| AMT_BALANCE | float | 当月信用卡余额 |
| AMT_CREDIT_LIMIT_ACTUAL | float | 当月信用卡额度 |
| AMT_DRAWINGS_ATM_CURRENT | float | 当月 ATM 取现金额 |
| AMT_DRAWINGS_CURRENT | float | 当月总取现金额 |
| AMT_DRAWINGS_OTHER_CURRENT | float | 当月其他取现金额 |
| AMT_DRAWINGS_POS_CURRENT | float | 当月 POS 消费金额 |
| AMT_INST_MIN_REGULARITY | float | 当月最低还款额 |
| AMT_PAYMENT_CURRENT | float | 当月还款金额 |
| AMT_PAYMENT_TOTAL_CURRENT | float | 当月总还款金额（含分期） |
| AMT_RECEIVABLE_PRINCIPAL | float | 应收本金 |
| AMT_RECIVABLE | float | 应收金额 |
| AMT_TOTAL_RECEIVABLE | float | 总应收金额（本金+利息） |
| CNT_DRAWINGS_ATM_CURRENT | int | 当月 ATM 取现次数 |
| CNT_DRAWINGS_CURRENT | int | 当月总取现次数 |
| CNT_DRAWINGS_OTHER_CURRENT | int | 当月其他取现次数 |
| CNT_DRAWINGS_POS_CURRENT | int | 当月 POS 消费次数 |
| CNT_INSTALMENT_MATURE_CUM | int | 已还分期数 |
| NAME_CONTRACT_STATUS | str | 合同状态：Active / Completed / Signed 等 |
| SK_DPD | int | 逾期天数（含宽限期） |
| SK_DPD_DEF | int | 逾期天数（不含宽限期） |

---

## 5. installments_payments（还款记录）

每行代表某笔贷款的某期还款记录。

| 字段 | 类型 | 含义 |
|------|------|------|
| SK_ID_PREV | int | 关联到 previous_application |
| SK_ID_CURR | int | 关联到主表 |
| NUM_INSTALMENT_VERSION | int | 还款计划版本（0=信用卡） |
| NUM_INSTALMENT_NUMBER | int | 第几期 |
| DAYS_INSTALMENT | int | 应还日期（距今天数，负值） |
| DAYS_ENTRY_PAYMENT | int | 实际还款日期（距今天数，负值） |
| AMT_INSTALMENT | float | 应还金额 |
| AMT_PAYMENT | float | 实际还款金额 |

---

## 6. POS_CASH_balance（POS/现金贷款月度状态）

每行代表某笔 POS 或现金贷款的某月状态。

| 字段 | 类型 | 含义 |
|------|------|------|
| SK_ID_PREV | int | 关联到 previous_application |
| SK_ID_CURR | int | 关联到主表 |
| MONTHS_BALANCE | int | 月份（负值） |
| CNT_INSTALMENT | float | 总分期数 |
| CNT_INSTALMENT_FUTURE | float | 剩余未还分期数 |
| NAME_CONTRACT_STATUS | str | 合同状态：Active / Completed / Signed 等 |
| SK_DPD | int | 逾期天数（含宽限期） |
| SK_DPD_DEF | int | 逾期天数（不含宽限期） |

---

## 7. previous_application（历史申请记录）

每行代表申请人在 Home Credit 的一次历史贷款申请。

| 字段 | 类型 | 含义 |
|------|------|------|
| SK_ID_PREV | int | 历史申请 ID |
| SK_ID_CURR | int | 关联到主表 |
| NAME_CONTRACT_TYPE | str | 贷款类型：Cash / Consumer / Revolving |
| AMT_ANNUITY | float | 年还款额 |
| AMT_APPLICATION | float | 申请金额 |
| AMT_CREDIT | float | 实际批准金额（可能 ≠ 申请金额） |
| AMT_DOWN_PAYMENT | float | 首付金额 |
| AMT_GOODS_PRICE | float | 商品价格 |
| WEEKDAY_APPR_PROCESS_START | str | 申请星期几 |
| HOUR_APPR_PROCESS_START | int | 申请时段 |
| FLAG_LAST_APPL_PER_CONTRACT | str | 是否为该合同的最后一次申请 |
| NFLAST_APPL_IN_DAY | int | 是否为当天最后一次申请 |
| RATE_DOWN_PAYMENT | float | 首付比例（归一化） |
| RATE_INTEREST_PRIMARY | float | 主利率（归一化） |
| RATE_INTEREST_PRIVILEGED | float | 优惠利率（归一化） |
| NAME_CASH_LOAN_PURPOSE | str | 现金贷款用途 |
| NAME_CONTRACT_STATUS | str | 审批结果：Approved / Refused / Canceled / Unused |
| DAYS_DECISION | int | 审批决定距今天数（负值） |
| NAME_PAYMENT_TYPE | str | 还款方式 |
| CODE_REJECT_REASON | str | 拒绝原因 |
| NAME_TYPE_SUITE | str | 陪同人员 |
| NAME_CLIENT_TYPE | str | 客户类型：New / Refreshed / Repeater / XNA |
| NAME_GOODS_CATEGORY | str | 商品类别 |
| NAME_PORTFOLIO | str | 贷款组合：POS / Cash / Cards / Car |
| NAME_PRODUCT_TYPE | str | 产品类型：x-sell / walk-in |
| CHANNEL_TYPE | str | 获客渠道 |
| SELLERPLACE_AREA | int | 销售区域 |
| NAME_SELLER_INDUSTRY | str | 卖方行业 |
| CNT_PAYMENT | int | 贷款期限（期数） |
| NAME_YIELD_GROUP | str | 利率分组：low / middle / high / XNA |
| PRODUCT_COMBINATION | str | 产品组合详情 |
| DAYS_FIRST_DRAWING | float | 首次放款距今天数 |
| DAYS_FIRST_DUE | float | 首次应还日期距今天数 |
| DAYS_LAST_DUE_1ST_VERSION | float | 最后应还日期（原始）距今天数 |
| DAYS_LAST_DUE | float | 最后应还日期（实际）距今天数 |
| DAYS_TERMINATION | float | 预计终止日期距今天数 |
| NFLAG_INSURED_ON_APPROVAL | float | 申请时是否要求保险 |
