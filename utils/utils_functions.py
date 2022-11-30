## UTILS TO USE OVER THE GIT REPOSITORY
## -------------------------------------------------------------------##
import os

##1. Function to create new folder
def folder_create(folder):
    """
    Function to check if a folder exist, otherwise, create one named like indicated
    :params: folder - name of the new folder 
    :return: 
    """
    if not os.path.exists(folder):
        os.makedirs(folder)