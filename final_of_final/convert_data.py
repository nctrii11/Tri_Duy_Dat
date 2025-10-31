import pandas as pd
import os # <-- Thêm thư viện 'os'

def chuyen_cot_cuoi_va_xuat_csv_vao_folder(df_input, ten_folder, ten_file):
    """
    Hàm này nhận DataFrame, chuyển cột cuối lên đầu,
    tạo folder nếu chưa có, và lưu file CSV vào đó.
    
    Args:
        df_input (pd.DataFrame): DataFrame gốc.
        ten_folder (str): Tên folder bạn muốn lưu vào (ví dụ: 'du_lieu_xuat').
        ten_file (str): Tên của file CSV (ví dụ: 'output.csv').
    """
    print(f"--- Đang xử lý: {ten_folder}/{ten_file} ---")

    # --- PHẦN MỚI: Xử lý folder ---
    # 1. Tạo đường dẫn file hoàn chỉnh
    # os.path.join sẽ tự động thêm dấu gạch chéo (/) hoặc (\)
    output_path = os.path.join(ten_folder, ten_file)
    
    # 2. Tạo thư mục nếu nó chưa tồn tại
    # exist_ok=True có nghĩa là sẽ không báo lỗi nếu folder đã có
    os.makedirs(ten_folder, exist_ok=True)
    print(f"Đã đảm bảo thư mục '{ten_folder}' tồn tại.")
    # -----------------------------

    # 3. Lấy danh sách các cột
    cols = df_input.columns.tolist()
    
    if not cols:
        print("Lỗi: DataFrame không có cột nào.")
        return

    # 4. Tạo danh sách cột mới
    thu_tu_cot_moi = [cols[-1]] + cols[:-1]
    
    # 5. Tạo DataFrame mới với thứ tự cột đã thay đổi
    df_output = df_input[thu_tu_cot_moi]
    
    # 6. Xuất DataFrame mới ra file CSV tại đường dẫn mới
    try:
        df_output.to_csv(output_path, index=False)
        print(f"Thành công! Đã lưu file vào đường dẫn:")
        print(f"==> {output_path}")
    
    except Exception as e:
        print(f"Lỗi khi đang lưu file: {e}")


ten_folder_moi = 'final_data'
ten_file_moi1 = 'final_data_train.csv'
ten_file_moi2 = 'final_data_test.csv'


df_input1 = pd.read_csv(r"D:\Workspace\Mon_Hoc\Mon_hoc\KPDL_TEAM\Tri_Duy_Dat\Data_clean\train_cleaned.csv")
df_input2 = pd.read_csv(r"D:\Workspace\Mon_Hoc\Mon_hoc\KPDL_TEAM\Tri_Duy_Dat\Data_clean\test_cleaned.csv")

chuyen_cot_cuoi_va_xuat_csv_vao_folder(df_input1, ten_folder_moi, ten_file_moi1)
chuyen_cot_cuoi_va_xuat_csv_vao_folder(df_input2, ten_folder_moi, ten_file_moi2)



