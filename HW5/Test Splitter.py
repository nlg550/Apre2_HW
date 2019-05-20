import os

# train_dir = os.path.join(os.curdir, "data\\train")
# test_dir = os.path.join(os.curdir, "data\\test")
# 
# for dir in os.listdir(train_dir):
#     i = 0
#     dir = os.path.join(train_dir, dir)
#     
#     for file in os.listdir(dir):
#         if i % 10 == 0:
#             if file.startswith("apple"): os.renames(os.path.join(dir, file), os.path.join(os.curdir, "data\\test\\apple\\" + file))
#             elif file.startswith("orange"): os.renames(os.path.join(dir, file), os.path.join(os.curdir, "data\\test\\orange\\" + file))
#             elif file.startswith("lemon"): os.renames(os.path.join(dir, file), os.path.join(os.curdir, "data\\test\\lemon\\" + file))
#             elif file.startswith("pear"): os.renames(os.path.join(dir, file), os.path.join(os.curdir, "data\\test\\pear\\" + file))
#         
#         i += 1
