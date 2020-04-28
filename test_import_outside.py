print('1st')
import uutils
print('2nd')
import uutils.utils as utils
print('3rd')

print('OUTSIDE THE PACKAGE BEING USED')
print(uutils)
print(utils)

# try:
#     # it seems you need the root pkg name to import if you are outside the package being import
#     # if you are inside it it seems you don't need to say which package you are using
#     import uutils
#     import uutils.utils as utils
#     import uutils.utils.logger as logger

#     print('OUTSIDE THE PACKAGE BEING USED')
#     print(uutils)
#     print(utils)
#     print(logger)
# except Exception as e:
#     print(e)
#     #print( sys.exc_info()[0] )
