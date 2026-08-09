# iPAS AI 中級應用規劃師｜科目 3：機器學習技術與應用
## 出題指南（命題規格版）

> 適用目標：iPAS AI 應用規劃師（中級）  
> 科目：科目 3－機器學習技術與應用  
> 用途：模擬試題、練習題庫、題組設計、考前測驗與能力檢核

---

## 一、考試基本規格

| 項目 | 規格 |
|---|---|
| 科目名稱 | 機器學習技術與應用 |
| 題型 | 單選題 |
| 題數 | 50 題 |
| 考試時間 | 90 分鐘 |
| 滿分 | 100 分 |
| 單科及格標準 | 70 分 |
| 程式相關題型 | 約占 25% |
| 程式題形式 | Python 基本語法、程式邏輯判斷、程式片段解析、題組型程式判讀 |

---

# 二、官方考試範圍架構

科目 3 可分為四大主題：

1. **L231 機器學習基礎數學**
2. **L232 機器學習與深度學習**
3. **L233 機器學習建模與參數調校**
4. **L234 機器學習治理**

---

# 三、各領域出題指南

## L231 機器學習基礎數學

### L23101 機率／統計之機器學習基礎應用

#### 建議命題內容

- 平均數、中位數、眾數
- 變異數、標準差
- 機率基本概念
- 條件機率
- 聯合機率
- 貝氏定理基本概念
- 常見機率分布
- 抽樣與統計推論
- 機率在分類問題中的應用

#### 建議題型

- 基本概念判斷
- 簡易計算題
- 機率結果解讀
- 分類模型機率輸出判讀
- 情境式統計判斷

#### 命題範例方向

例如：

> 某二元分類模型輸出預測機率為 0.8，若分類門檻為 0.5，模型會將該筆資料判定為哪一類？

---

## L23102 線性代數之機器學習基礎應用

### 建議命題內容

- Scalar、Vector、Matrix
- 向量維度
- 矩陣維度
- 矩陣加法
- 矩陣乘法
- 轉置矩陣
- 向量內積
- 線性轉換
- 神經網路權重矩陣維度
- Batch × Feature 的資料表示方式

### 建議題型

- 維度判讀
- 矩陣是否可相乘
- 輸入輸出 Shape 判斷
- 神經網路 Linear Layer 維度題
- NumPy 陣列 Shape 程式題

### 命題重點

避免過度偏向高等線性代數證明，應以：

> 「機器學習模型實際會用到的矩陣概念」

作為出題核心。

---

## L23103 數值優化技術與方法

### 建議命題內容

- 微分
- 偏微分
- Gradient
- Loss Function
- Cost Function
- Gradient Descent
- Learning Rate
- Parameter Update
- Local Minimum
- Optimization
- 收斂
- 學習率過高或過低的影響

### 必考公式概念

\[
w_{new} = w - \eta \frac{\partial L}{\partial w}
\]

其中：

- \(w\)：模型參數
- \(\eta\)：Learning Rate
- \(L\)：Loss Function

### 建議題型

- 一步梯度下降計算
- 學習率判斷
- Loss 變化判斷
- 梯度方向判斷
- 模型無法收斂的原因判斷

---

# L232 機器學習與深度學習

## L23201 機器學習原理與技術

### 建議命題內容

- Supervised Learning
- Unsupervised Learning
- Semi-Supervised Learning
- Classification
- Regression
- Clustering
- Training
- Inference
- Label
- Feature
- Target
- 模型泛化能力
- Bias 與 Variance

### 必須能判斷

| 問題類型 | 建議方法 |
|---|---|
| 預測房價 | Regression |
| 垃圾郵件辨識 | Classification |
| 客戶分群 | Clustering |
| 信用風險判斷 | Classification |
| 銷售額預測 | Regression |
| 未標記客戶分群 | Unsupervised Learning |

### 建議題型

以「情境判斷」為主，不宜只考名詞定義。

---

## L23202 常見機器學習演算法

### 建議命題演算法

#### Regression

- Linear Regression

#### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN

#### Clustering

- K-Means

#### 其他常見方法

- Naive Bayes
- Ensemble Learning
- Boosting 基本概念

### 建議命題方向

考生應能判斷：

1. 模型適合處理什麼問題
2. 模型主要特性
3. 優點與限制
4. 哪些模型需要特徵縮放
5. 哪些模型較容易解釋
6. 哪些模型可處理非線性問題

### 典型情境

> 若希望模型具有較好的可解釋性，且需呈現決策規則，可優先考慮 Decision Tree。

---

## L23203 深度學習原理與框架

### 建議命題內容

- Artificial Neural Network
- Neuron
- Weight
- Bias
- Activation Function
- Forward Propagation
- Backpropagation
- Epoch
- Batch
- Batch Size
- CNN
- RNN
- LSTM
- Transfer Learning
- Fine-tuning
- Pretrained Model

### Activation Function

建議涵蓋：

- ReLU
- Sigmoid
- Softmax
- Tanh 基本概念

### 模型應用

| 模型 | 常見應用 |
|---|---|
| CNN | 影像辨識 |
| RNN | 序列資料 |
| LSTM | 長期序列依賴 |
| Transfer Learning | 小型資料集、降低訓練成本 |

### 深度學習框架

可涵蓋基本用途判讀：

- PyTorch
- TensorFlow
- Keras

不宜要求過度偏向特定框架 API 細節。

---

# L233 機器學習建模與參數調校

## L23301 數據準備與特徵工程

### 建議命題內容

- Missing Value
- Outlier
- Duplicate Data
- Data Cleaning
- Feature Engineering
- Feature Selection
- One-Hot Encoding
- Label Encoding
- Normalization
- Standardization
- 資料不平衡
- Data Leakage
- Train / Validation / Test Split

### 出題重點

考生需理解完整資料流程：

```text
資料取得
  ↓
資料清理
  ↓
缺失值處理
  ↓
類別編碼
  ↓
特徵縮放
  ↓
特徵工程
  ↓
資料切分
  ↓
模型訓練
```

### Data Leakage

應列為高重要度考點。

典型題目：

> 若先用完整資料計算 StandardScaler 的平均值與標準差，再進行 Cross Validation，可能造成什麼問題？

正確概念：

> 可能發生 Data Leakage。

---

## L23302 模型選擇與架構設計

### 建議命題內容

- Problem Definition
- Classification / Regression 選擇
- 模型複雜度
- Bias
- Variance
- Underfitting
- Overfitting
- 模型解釋性
- 計算成本
- 資料量
- 特徵數量

### 建議情境題

例如：

> 某模型 Training Accuracy 99%，Validation Accuracy 72%，最可能發生什麼問題？

應判定為：

> Overfitting。

---

## L23303 模型訓練、評估與驗證

### 建議命題內容

- Training Set
- Validation Set
- Test Set
- Hold-out Validation
- Cross Validation
- K-Fold
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC
- AUC
- Regression 評估指標基本概念

### Confusion Matrix

建議熟悉：

- TP：True Positive
- TN：True Negative
- FP：False Positive
- FN：False Negative

### 必考公式

#### Accuracy

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

#### Precision

\[
Precision = \frac{TP}{TP+FP}
\]

#### Recall

\[
Recall = \frac{TP}{TP+FN}
\]

#### F1-score

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision+Recall}
\]

### 情境判斷

#### 偏重 Recall

適合：

- 癌症篩檢
- 詐欺偵測
- 異常事件偵測

核心概念：

> 真正有問題的案例不要漏掉。

#### 偏重 Precision

適合：

- 垃圾郵件過濾
- 高成本人工複審
- 誤報成本較高的系統

核心概念：

> 被模型判定為正例者，要盡量是真的正例。

---

## L23304 模型調整與優化

### 建議命題內容

- Hyperparameter
- Parameter
- Grid Search
- Random Search
- Cross Validation
- Regularization
- L1
- L2
- Dropout
- Early Stopping
- Learning Rate
- Batch Size
- Epoch
- 過度擬合改善方法
- 欠擬合改善方法

### 典型命題

#### Overfitting 改善

可能包含：

- 增加訓練資料
- Data Augmentation
- Regularization
- Dropout
- Early Stopping
- 降低模型複雜度

#### Underfitting 改善

可能包含：

- 增加模型複雜度
- 增加有效特徵
- 延長適當訓練
- 降低過強的 Regularization

---

# L234 機器學習治理

## L23401 數據隱私、安全與合規

### 建議命題內容

- 個人資料
- 敏感資料
- Data Privacy
- Data Security
- Access Control
- Data Minimization
- 去識別化
- 匿名化
- 資料生命週期
- 資料存取權限
- 訓練資料合法性
- 模型輸入資料安全
- 合規性概念

### 建議題型

以實際企業 AI 導入情境為主。

例如：

> 建立醫療模型時，若使用病患資料進行訓練，最先應注意什麼？

應考量：

- 資料使用是否合法
- 隱私
- 權限
- 去識別化
- 安全控管

---

## L23402 演算法偏見與公平性

### 建議命題內容

- Algorithmic Bias
- Data Bias
- Selection Bias
- Historical Bias
- Fairness
- 群體差異
- 模型公平性評估
- 模型可解釋性
- AI Responsible Use

### 情境命題

例如：

> 貸款模型對兩個條件相近的族群產生大幅不同的核准率，應如何處理？

正確命題方向：

- 檢查資料是否有偏差
- 檢查 Feature
- 評估不同群體模型結果
- 使用公平性指標
- 必要時重新訓練或調整模型

避免錯誤選項：

- 只看 Accuracy
- 刪除測試資料
- 增加模型層數即可解決
- 忽略群體差異

---

# 四、程式相關題型出題指南

正式練習題庫中，建議約 **25% 為程式相關題型**。

若模擬正式 50 題：

> 建議安排約 12～13 題 Python／程式判讀題。

---

## 1. Python 基礎語法

建議涵蓋：

- List
- Tuple
- Dictionary
- Loop
- If
- Function
- Index
- Slice
- 基本型別
- 基本運算

例如：

```python
x = [10, 20, 30, 40]
print(x[1:3])
```

考：

> 輸出結果為何？

---

## 2. NumPy

建議涵蓋：

- np.array
- shape
- reshape
- mean
- sum
- axis
- matrix operation

例如：

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x.shape)
```

---

## 3. Pandas

建議涵蓋：

- DataFrame
- 欄位選取
- Missing Value
- dropna
- fillna
- mean
- describe
- 基本資料篩選

---

## 4. Scikit-learn

建議涵蓋：

```python
train_test_split()
```

```python
model.fit()
```

```python
model.predict()
```

```python
accuracy_score()
```

```python
StandardScaler()
```

```python
cross_val_score()
```

### 命題應要求考生理解

```python
model.fit(X_train, y_train)
```

代表：

> 模型訓練

而：

```python
model.predict(X_test)
```

代表：

> 使用模型進行推論。

---

## 5. PyTorch／深度學習程式判讀

建議出題層級以基本概念為主。

例如：

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)

x = torch.randn(10, 4)
y = model(x)

print(y.shape)
```

應能判讀：

```text
torch.Size([10, 2])
```

命題重點：

- Batch Size
- Input Features
- Output Features
- Tensor Shape
- Linear Layer

---

# 五、建議命題比例

若製作 **50 題正式模擬測驗**，建議分配如下：

| 範圍 | 建議占比 | 50 題配置 |
|---|---:|---:|
| L231 機器學習基礎數學 | 20% | 10 題 |
| L232 機器學習與深度學習 | 30% | 15 題 |
| L233 建模與參數調校 | 35% | 17～18 題 |
| L234 機器學習治理 | 15% | 7～8 題 |

其中：

> 約 12～13 題應包含 Python 或程式碼判讀。

---

# 六、建議難度比例

為模擬正式考試，建議：

| 難度 | 比例 | 50 題配置 |
|---|---:|---:|
| 基礎題 | 30% | 約 15 題 |
| 中等題 | 50% | 約 25 題 |
| 進階題 | 20% | 約 10 題 |

---

## 基礎題

主要測驗：

- 名詞定義
- 基本概念
- 簡單公式
- 基本程式輸出

例如：

> 下列何者屬於監督式學習？

---

## 中等題

主要測驗：

- 情境判斷
- 模型選擇
- 評估指標選擇
- 資料處理流程
- 程式碼理解

例如：

> 詐欺偵測模型中，若希望降低漏抓真正詐欺交易的比例，應優先改善哪項指標？

---

## 進階題

主要測驗：

- 多觀念整合
- Data Leakage
- 模型調參
- Overfitting
- Cross Validation
- 多步驟程式碼判讀

例如：

> 在 K-Fold Cross Validation 前直接對全部資料進行 StandardScaler.fit()，此流程可能造成什麼問題？

---

# 七、建議題型比例

50 題模擬考可以再細分：

| 題型 | 建議比例 |
|---|---:|
| 基本概念題 | 20% |
| 情境判斷題 | 30% |
| 模型／演算法選擇題 | 15% |
| 計算題 | 10% |
| Python／程式判讀 | 25% |

實際命題時，程式題亦可同時屬於：

- 特徵工程
- 模型訓練
- 評估
- 深度學習

因此分類不必完全互斥。

---

# 八、選項設計原則

每題四個選項：

```text
(A)
(B)
(C)
(D)
```

建議：

- 只有一個最佳答案
- 避免兩個選項均可能成立
- 錯誤選項應具干擾性
- 不宜使用明顯荒謬答案
- 避免「以上皆是」與「以上皆非」
- 避免單純靠選項長度猜答案
- 正確答案位置應平均分布

---

# 九、情境題設計原則

中級考試應避免大量純記憶題。

建議結構：

```text
背景
↓
問題
↓
限制條件
↓
要求考生選擇最佳方法
```

例如：

> 某工廠擁有 2,000 張已標記的瑕疵產品圖片，但 GPU 資源有限，希望快速建立影像分類模型。下列何種策略最適合？

命題概念：

> Transfer Learning。

---

# 十、常見陷阱題設計

## 1. Accuracy 陷阱

不平衡資料：

```text
正常：99%
異常：1%
```

若全部預測為正常：

```text
Accuracy = 99%
```

但模型並沒有實際偵測能力。

因此可考：

- Precision
- Recall
- F1
- Confusion Matrix

---

## 2. Data Leakage 陷阱

錯誤：

```text
全部資料
↓
StandardScaler.fit()
↓
Train/Test Split
```

較佳：

```text
Train/Test Split
↓
Scaler.fit(X_train)
↓
Scaler.transform(X_train)
↓
Scaler.transform(X_test)
```

---

## 3. Overfitting 陷阱

```text
Training Accuracy ↑
Validation Accuracy ↓
```

應判斷為：

> Overfitting。

---

## 4. Learning Rate 陷阱

過高：

> Loss 震盪甚至發散。

過低：

> 收斂非常慢。

---

## 5. Precision／Recall 陷阱

Recall：

> 找出多少真正 Positive。

Precision：

> 預測為 Positive 的案例中有多少是真的。

---

# 十一、出題時應避免的內容

除非官方範圍進一步擴充，不建議把大量題目放在：

- 高階微積分證明
- 高等機率論證明
- 深度 PyTorch API 細節
- CUDA 程式設計
- Transformer 原始論文數學推導
- 自行實作 Backpropagation
- 複雜 TensorFlow 底層 API
- 過度偏向特定模型套件版本

科目定位仍應為：

> **AI 應用規劃與機器學習實務能力**

而非：

> 純資料科學家或深度學習研究員考試。

---

# 十二、50 題正式模擬考建議藍圖

可依下列配置製作完整模擬考：

## L231：10 題

- 機率／統計：4 題
- 線性代數：3 題
- 數值優化：3 題

## L232：15 題

- ML 原理：5 題
- 常見 ML 演算法：6 題
- 深度學習：4 題

## L233：18 題

- 資料準備與特徵工程：5 題
- 模型選擇：3 題
- 模型評估與驗證：6 題
- 模型調整與優化：4 題

## L234：7 題

- 數據隱私、安全與合規：3 題
- 演算法偏見與公平性：4 題

總計：

```text
10 + 15 + 18 + 7 = 50 題
```

其中至少：

```text
12～13 題
```

應設計為 Python／程式判讀相關題型。

---

# 十三、單次 10 題練習模組建議配置

若每次產生 10 題，可採：

| 類型 | 題數 |
|---|---:|
| L231 | 2 |
| L232 | 3 |
| L233 | 3～4 |
| L234 | 1～2 |
| 其中程式題 | 2～3 |

如此連續產生 5 回，即可形成約 50 題完整模擬題庫。

---

# 十四、命題品質檢核表

每完成一道題目後，可檢查：

- [ ] 是否屬於官方科目 3 範圍
- [ ] 是否只有一個最佳答案
- [ ] 錯誤選項是否具有合理干擾性
- [ ] 是否避免純死背
- [ ] 是否具有機器學習應用情境
- [ ] 計算量是否適合考試時間
- [ ] 程式碼是否能由閱讀推導答案
- [ ] 是否避免需要實際執行程式才能作答
- [ ] 專有名詞是否正確
- [ ] 中文描述是否沒有歧義
- [ ] 是否符合中級難度
- [ ] 正確答案位置是否平均分散

---

# 十五、建議出題 Prompt

若後續使用 AI 建立題庫，可以使用：

```text
你是一位 iPAS AI 應用規劃師（中級）命題委員。

請依「科目 3：機器學習技術與應用」考試範圍，
產生符合正式考試風格的單選題。

命題範圍：
L231 機器學習基礎數學
L232 機器學習與深度學習
L233 機器學習建模與參數調校
L234 機器學習治理

規則：
1. 每題四個選項 A、B、C、D。
2. 每題只有一個最佳答案。
3. 以實務情境題為主要題型。
4. 避免單純名詞背誦。
5. 約 25% 題目為 Python 或程式碼判讀。
6. 程式題以 NumPy、Pandas、Scikit-learn、PyTorch 基礎操作為主。
7. 程式碼必須可由閱讀判斷答案，不要求實際執行。
8. 包含 Accuracy、Precision、Recall、F1、Cross Validation、
   Data Leakage、Overfitting、Underfitting 等核心概念。
9. 題目難度比例：
   - 基礎 30%
   - 中等 50%
   - 進階 20%
10. 正確答案 A、B、C、D 應平均分布。
11. 題目與答案分開輸出。
12. 題目描述不得洩漏答案。
```

---

# 十六、核心命題原則摘要

科目 3 的最佳出題方向不是：

> 「你記得多少機器學習名詞？」

而是：

> **「你能不能根據資料、模型與應用情境，選擇正確的機器學習方法。」**

因此題庫應優先測驗：

1. **看得懂問題**
2. **知道選什麼模型**
3. **知道怎麼準備資料**
4. **知道怎麼訓練模型**
5. **知道怎麼評估模型**
6. **知道模型出問題時如何改善**
7. **看得懂基本 Python／ML 程式碼**
8. **知道 AI 模型在隱私、公平與安全上的風險**

這八項能力可作為整份科目 3 模擬題庫的核心命題準則。
