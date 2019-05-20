import os

# Move to the corrrect folder and merge multiple types of the same fruit
# base_dir = os.path.curdir
#  
# i = 0;
# j = 0;
# k = 0;
# w = 0;
#  
# train_dir = os.path.join(base_dir, 'Dataset\\Training')    
# for dir in os.listdir(train_dir):
#     if(dir.startswith("Apple")):
#         dir = os.path.join(train_dir, dir)    
#          
#         for filename in os.listdir(dir):
#             filename = os.path.join(dir, filename)
#              
#             os.rename(filename, os.path.join(base_dir, "data\\\\training\\\\apple\\\\apple_" + str(i) + ".jpg"))
#             i += 1
#              
#     elif(dir.startswith("Lemon")):
#         dir = os.path.join(train_dir, dir)    
#                   
#         for filename in os.listdir(dir):
#             filename = os.path.join(dir, filename)
#             os.rename(filename, os.path.join(base_dir, "data\\training\\lemon\\lemon_" + str(j) + ".jpg"))
#             j += 1
#               
#     elif(dir.startswith("Orange")):
#         dir = os.path.join(train_dir, dir)    
#                   
#         for filename in os.listdir(dir):
#             filename = os.path.join(dir, filename)
#             os.rename(filename, os.path.join(base_dir, "data\\training\\orange\\orange_" + str(k) + ".jpg"))
#             k += 1
#               
#     elif(dir.startswith("Pear")):
#         dir = os.path.join(train_dir, dir)    
#                   
#         for filename in os.listdir(dir):
#             filename = os.path.join(dir, filename)
#             os.rename(filename, os.path.join(base_dir, "data\\training\\pear\\pear_" + str(w) + ".jpg"))
#             w += 1
#  
# i = 0;
# j = 0;
# k = 0;
# w = 0;
#  
# test_dir = os.path.join(base_dir, 'Dataset\\Test')    
# for dir in os.listdir(test_dir):
#     if(dir.startswith("Apple")):
#         dir = os.path.join(test_dir, dir)    
#          
#         for filename in os.listdir(dir):
#             filename = os.path.join(dir, filename)            
#             os.rename(filename, os.path.join(base_dir, "data\\test\\apple\\apple_" + str(i) + ".jpg"))
#             i += 1
#              
#     elif(dir.startswith("Lemon")):
#         dir = os.path.join(test_dir, dir)    
#                   
#         for filename in os.listdir(dir):
#             filename = os.path.join(dir, filename)
#             os.rename(filename, os.path.join(base_dir, "data\\test\\lemon\\lemon_" + str(j) + ".jpg"))
#             j += 1
#               
#     elif(dir.startswith("Orange")):
#         dir = os.path.join(test_dir, dir)    
#                   
#         for filename in os.listdir(dir):
#             filename = os.path.join(dir, filename)
#             os.rename(filename, os.path.join(base_dir, "data\\test\\orange\\orange_" + str(k) + ".jpg"))
#             k += 1
#               
#     elif(dir.startswith("Pear")):
#         dir = os.path.join(test_dir, dir)    
#                   
#         for filename in os.listdir(dir):
#             filename = os.path.join(dir, filename)
#             os.rename(filename, os.path.join(base_dir, "data\\test\\pear\\pear_" + str(w) + ".jpg"))
#             w += 1
