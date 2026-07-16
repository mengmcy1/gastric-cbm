"""少量患者 SAE 演示入口，实际流程复用正式脚本。"""

import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORMAL_DIR = os.path.join(os.path.dirname(BASE_DIR), '正式代码')
sys.path.insert(0, FORMAL_DIR)

from sae_discovery import main


if __name__ == '__main__':
    main(['--demo', *sys.argv[1:]])
