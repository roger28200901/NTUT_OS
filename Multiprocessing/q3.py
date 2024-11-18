import time
import numpy as np
from multiprocessing import Process, Array

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

def multiply_row_range(start_row, end_row, A, B, C_shared, rows_C, cols_C):
    """計算指定行範圍的矩陣乘法，並將結果寫入共享記憶體"""
    cols_B = B.shape[1]
    cols_A = A.shape[1]
    C = np.frombuffer(C_shared.get_obj(), dtype=complex).reshape((rows_C, cols_C))

    for i in range(start_row, end_row):
        for j in range(cols_B):
            sum_val = 0
            for k in range(cols_A):
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val

def matrix_multiply_multiprocessing(A, B, num_processes):
    """使用多處理程序進行矩陣乘法"""
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    if cols_A != rows_B:
        raise ValueError("矩陣維度不符合乘法要求")

    # 初始化共享記憶體的結果矩陣
    C_shared = Array('d', rows_A * cols_B * 2)  # 每個複數需要 2 個 double (實部和虛部)

    # 每個處理程序處理的行數
    rows_per_process = rows_A // num_processes
    processes = []

    # 建立處理程序
    for i in range(num_processes):
        start_row = i * rows_per_process
        end_row = start_row + rows_per_process if i < num_processes - 1 else rows_A

        process = Process(target=multiply_row_range, args=(start_row, end_row, A, B, C_shared, rows_A, cols_B))
        processes.append(process)
        process.start()

    # 等待所有處理程序完成
    for process in processes:
        process.join()

    # 將共享記憶體結果轉換為 NumPy 矩陣
    C = np.frombuffer(C_shared.get_obj(), dtype=complex).reshape((rows_A, cols_B))
    return C

def main():
    # 矩陣維度
    rows_A, cols_A = 500, 500
    rows_B, cols_B = 500, 500

    # 初始化矩陣 A 和 B
    A = initialize_matrix_A(rows_A, cols_A)
    B = initialize_matrix_B(rows_B, cols_B)

    # 設定處理程序數量（如 10 或 50）
    NUM_PROCESSES = 10  # 可以修改為 50 測試不同情況

    # 計時開始
    start_time = time.time()

    # 執行多處理程序矩陣乘法
    C = matrix_multiply_multiprocessing(A, B, NUM_PROCESSES)

    # 計時結束
    end_time = time.time()

    # 輸出執行時間
    execution_time = (end_time - start_time) * 1000  # 毫秒
    print(f"執行時間 (multiprocessing, {NUM_PROCESSES} processes): {execution_time:.2f} ms")

    # 驗證結果矩陣 C
    print("結果矩陣 C 的部分內容:")
    print(C[:5, :5])  # 印出部分內容確認

if __name__ == "__main__":
    main()
