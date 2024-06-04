import os
from os import listdir
from os.path import isfile, join
import re


def get_all_files_in_dir(root_dir, recursive, path_filter = None):
    """
    Returns all the files in a directory. For each file it returns the relative path to the given dir

    Parameters
    ----------
    root_dir: str,
        path to the directory we want to get the file list

    recursive: bool,
        if True files are searched recursively in a folder

    Returns
    -------

    """
    file_paths = []
    if recursive:
        for root, subFolders, files in os.walk(root_dir):
            for file in files:
                new_file = join(root, file)
                if not path_filter is None:
                    if not bool(re.search(path_filter, new_file)):
                        continue
                file_paths.append(new_file)
    else:
        file_paths = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]

    return file_paths
