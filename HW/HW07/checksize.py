from PIL import Image
import os

def get_pgm_size(file_path):
    """
    獲取 .pgm 圖像的大小。
    
    參數：
    file_path (str): 圖像文件的路徑。
    
    返回：
    (width, height) (tuple): 圖像的寬度和高度。
    """
    with Image.open(file_path) as img:
        return img.size

def main():
    directory = './Yale_Face_Database/Yale_Face_Database/Training'
    pgm_files = [file.path for file in os.scandir(directory) if file.path.endswith('.pgm') and file.is_file()]
    
    for file_path in pgm_files:
        width, height = get_pgm_size(file_path)
        print(f'{file_path}: {width}x{height}')

if __name__ == '__main__':
    main()
