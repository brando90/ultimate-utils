import sys

def print_sys_path():
    for path in sys.path:
        print(path)

def path():
    import sys
    [print(p) for p in sys.path]

if __name__ == '__main__':
    # if the function is run as main (not just imported), then show the sys path
    print_sys_path()