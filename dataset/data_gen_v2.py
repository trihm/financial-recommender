import pandas as pd
import numpy as np

# Số lượng người dùng giả lập
num_users = 1000

# Tạo dữ liệu ngẫu nhiên
data = {
    'user_id': range(1, num_users + 1),
    'income': np.random.randint(10, 50, size=num_users) * 1000000,
    'total_spending_ratio': np.random.uniform(0.3, 1.0, size=num_users),
    'is_budget_tracker_user': np.random.randint(0, 2, size=num_users)
}

df = pd.DataFrame(data)

# Tính tổng chi tiêu
df['total_spending'] = df['income'] * df['total_spending_ratio']

# Phân loại hành vi chi tiêu
def classify_behavior(ratio):
    if ratio < 0.5:
        return 'savings_oriented'
    elif 0.5 <= ratio <= 0.7:
        return 'balanced'
    else:
        return 'high_spender'

df['spending_behavior_class'] = df['total_spending_ratio'].apply(classify_behavior)

# Tạo dữ liệu chi tiêu theo các danh mục cụ thể
# Lưu ý: Tổng các tỷ lệ này phải xấp xỉ 1
df['spending_food_ratio'] = np.random.uniform(0.2, 0.4, size=num_users)
df['spending_transport_ratio'] = np.random.uniform(0.05, 0.15, size=num_users)
df['spending_bills_ratio'] = np.random.uniform(0.1, 0.2, size=num_users)
df['spending_entertainment_ratio'] = np.random.uniform(0.05, 0.15, size=num_users)
df['spending_shopping_ratio'] = np.random.uniform(0.05, 0.15, size=num_users)
df['spending_education_ratio'] = np.random.uniform(0.05, 0.1, size=num_users)

# Tính toán số tiền chi tiêu thực tế cho từng danh mục
df['spending_food'] = df['spending_food_ratio'] * df['total_spending']
df['spending_transport'] = df['spending_transport_ratio'] * df['total_spending']
df['spending_bills'] = df['spending_bills_ratio'] * df['total_spending']
df['spending_entertainment'] = df['spending_entertainment_ratio'] * df['total_spending']
df['spending_shopping'] = df['spending_shopping_ratio'] * df['total_spending']
df['spending_education'] = df['spending_education_ratio'] * df['total_spending']

# Tính toán cột tiết kiệm và đầu tư
# Đây là phần còn lại sau khi đã chi tiêu
df['spending_investment_savings'] = df['income'] - df['total_spending']

# Xuất DataFrame ra file CSV
df.to_csv('spending_data_detailed.csv', index=False)

print("Đã xuất dữ liệu chi tiết thành công ra file 'spending_data_detailed.csv'")
