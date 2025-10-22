# 載入所需套件
library(C50)
library(caret)

# 讀取資料
x_train <- read.csv("processed_data/x_train.csv")
x_test <- read.csv("processed_data/x_test.csv")
y_train <- read.csv("processed_data/y_train.csv")$income
y_test <- read.csv("processed_data/y_test.csv")$income

# 建立 C5.0 模型
c50_model <- C5.0(x = x_train, y = as.factor(y_train))

# 進行預測
train_predictions <- predict(c50_model, x_train)  # 訓練集預測
test_predictions <- predict(c50_model, x_test)    # 測試集預測

# 計算測試集各項指標
confusion_matrix <- confusionMatrix(test_predictions, as.factor(y_test))

# 提取指標
accuracy <- confusion_matrix$overall['Accuracy']
precision <- confusion_matrix$byClass['Pos Pred Value']  # Precision
recall <- confusion_matrix$byClass['Sensitivity']        # Recall
f1_score <- confusion_matrix$byClass['F1']               # F1-score

# 顯示所有指標
cat("C5.0 測試集評估指標:\n")
cat("Accuracy :", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall   :", round(recall, 4), "\n")
cat("F1-score :", round(f1_score, 4), "\n")

# 計算訓練集準確率
train_accuracy <- mean(train_predictions == as.factor(y_train))
cat("C5.0 訓練集準確率:", round(train_accuracy, 4), "\n")

# 建立包含所有預測結果的資料框
results_df <- data.frame(
  actual = y_test,
  predicted = as.numeric(as.character(test_predictions))
)

# 儲存所有預測結果到 CSV
write.csv(results_df, "c50_result.csv", row.names = FALSE)

# 儲存所有評估指標
metrics_df <- data.frame(
  metric = c("accuracy", "precision", "recall", "f1_score", "train_accuracy"),
  value = c(accuracy, precision, recall, f1_score, train_accuracy)
)
write.csv(metrics_df, "c50_metrics.csv", row.names = FALSE)

print("所有預測結果已保存到 c50_result.csv")
print("評估指標已保存到 c50_metrics.csv")
