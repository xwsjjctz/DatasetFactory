import os
from natsort import natsorted, ns

slash = os.path.sep                                                 # 系统文件路径分隔符
CUDALIST = 0                                                        # 选择gpu

# 返回目录所有文件的列表
def file_path(filepath):
    for path in os.walk(filepath):
        path = natsorted(path[2], alg=ns.PATH)
        return path