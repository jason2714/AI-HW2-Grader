HW2的批改程式已經寫完
可以在eric幫忙開的空間中的
/home/hw2中找到

前置作業如下:
1.上傳的.py放到student code資料夾內(格式要對- r12345678_hw2.py)。
2.整理好學號寫在student_list.txt裡面，一行一筆學號。
3.若使用windows平台操作，code中遇到while true會出錯，盡量使用linux

批次批改指令:
python3 grader.py
optional arguments:
  -h, --help            show this help message and exit
  -C, --code_dir CODE_DIR
      default: student_code/
      Directory to Student Code.
  -IL, --list_path LIST_PATH
      default: student_list.txt
      Path to Student List.
  -I, --ID_list [ID_LIST ...]
      default: No Use
      Student ID List.
  -L, --log_dir LOG_DIR
      default: student_log/
      Directory to save the log file.
  -G, --grade_path GRADE_PATH
      default: grade.csv
      Path to Grade file.
  -M, --modify_grade
      action: store_true    
      default: false
      modify exist csv or create csv


只judge數份作業指令:
python3 grader.py -I r12345678 r87654321 ...

不會蓋掉原本的args.grade_path
會跟args.grade_path中原本的成績比較
分數變高或是沒出現過才會更新

輸出:
1.成績會寫入args.grade_path
每一列的資料如下
['Student ID', 'Problem1', 'Problem2', 'Problem3', 'Problem4', 'Total points']

滿分為20, 30, 30, 30, 110
如果未繳交作業total potins會顯示為-1分

學生的log會存在args.log_dir中
可以查看問題出在哪裡