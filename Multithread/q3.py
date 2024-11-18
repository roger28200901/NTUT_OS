import time
import numpy as np
from threading import Thread
import threading

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

def multiply_row_range(start_row, end_row, A, B, C):
    """計算指定行範圍的矩陣乘法"""
    cols_B = B.shape[1]
    cols_A = A.shape[1]

    for i in range(start_row, end_row):
        for j in range(cols_B):
            sum_val = 0
            for k in range(cols_A):
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val

def matrix_multiply_multithreading(A, B, num_threads):
    """使用多執行緒進行矩陣乘法"""
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError("矩陣維度不符合乘法要求")

    # 初始化結果矩陣
    C = np.zeros((rows_A, cols_B), dtype=complex)

    # 每個執行緒處理的行數
    rows_per_thread = rows_A // num_threads
    threads = []

    # 建立執行緒
    for i in range(num_threads):
        start_row = i * rows_per_thread
        end_row = start_row + rows_per_thread if i < num_threads - 1 else rows_A

        thread = Thread(target=multiply_row_range, args=(start_row, end_row, A, B, C))
        threads.append(thread)
        thread.start()

    # 等待所有執行緒完成
    for thread in threads:
        thread.join()

    return C

def main():
    # 矩陣維度
    rows_A, cols_A = 500, 500
    rows_B, cols_B = 500, 500

    # 初始化矩陣 A 和 B
    A = initialize_matrix_A(rows_A, cols_A)
    B = initialize_matrix_B(rows_B, cols_B)

    # 設定執行緒數量（如 10 或 50）
    NUM_THREADS = 10  # 可以修改為 50 測試不同情況

    # 計時開始
    start_time = time.time()

    # 執行多執行緒矩陣乘法
    C = matrix_multiply_multithreading(A, B, NUM_THREADS)

    # 計時結束
    end_time = time.time()

    # 輸出執行時間
    execution_time = (end_time - start_time) * 1000  # 毫秒
    print(f"執行時間 (multithreading, {NUM_THREADS} threads): {execution_time:.2f} ms")

    # 驗證結果矩陣 C
    print("結果矩陣 C 的部分內容:")
    print(C[:5, :5])  # 印出部分內容確認

if __name__ == "__main__":
    main()