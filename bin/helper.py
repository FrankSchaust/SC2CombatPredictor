from bin.util import *
from lib.config import REPO_DIR
def main():
    file_path = os.path.join(REPO_DIR, '1_3a', 'all_csv_from_version_1_3a.csv')
    print(read_summed_up_csv(file_path))
    
 
if __name__ == "__main__":
    main()