import time
import numpy as np

def initialize_matrix_A(rows, cols):
    """初始化矩陣 A，元素公式為 a[i,j] = 6.5*i - 1.8*j"""
    matrix = np.zeros((rows, cols), dtype=complex)
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = complex(6.5 * i, -1.8 * j)
    return matrix

def initialize_matrix_B(rows, cols):
    """初始化矩陣 B，元素公式為 b[i,j] = (30 + 5.5*i) - 12.1j"""
    matrix = np.zeros((rows, cols), dtype=complex)
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = complex(30 + 5.5 * i, -12.1)
    return matrix

def matrix_multiply(A, B):
    """使用 for loop 計算矩陣乘法"""
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError("矩陣維度不符合乘法要求")

    # 初始化結果矩陣
    C = np.zeros((rows_A, cols_B), dtype=complex)

    # 矩陣乘法
    for i in range(rows_A):
        for j in range(cols_B):
            sum_val = 0
            for k in range(cols_A):
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val

    return C

def main():
    # 矩陣維度
    rows_A, cols_A = 500, 500
    rows_B, cols_B = 500, 500

    # 初始化矩陣 A 和 B
    A = initialize_matrix_A(rows_A, cols_A)
    B = initialize_matrix_B(rows_B, cols_B)

    # 計時開始
    start_time = time.time()

    # 矩陣乘法
    C = matrix_multiply(A, B)

    # 計時結束
    end_time = time.time()

    # 輸出執行時間
    execution_time = (end_time - start_time) * 1000  # 毫秒
    print(f"執行時間 (for loop): {execution_time:.2f} ms")

    # 驗證結果矩陣 C
    print("結果矩陣 C 的部分內容:")
    print(C[:5, :5])  # 印出部分內容確認

if __name__ == "__main__":
    main()
