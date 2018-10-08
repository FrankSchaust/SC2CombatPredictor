from bin.util import *
from lib.config import REPO_DIR
def main():

        
    replay_parsed_files = []
    replay_parsed_files = build_file_array(version='1_3a')
    print(len(filter_close_matchups(replay_parsed_files)))
    
 
if __name__ == "__main__":
    main()