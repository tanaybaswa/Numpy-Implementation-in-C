from distutils.core import setup, Extension
import sysconfig

def main():
    CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
    LDFLAGS = ['-fopenmp']
    # Use the setup function we imported and set up the modules.
    # You may find this reference helpful: https://docs.python.org/3.6/extending/building.html
    module1 = Extension('numc', sources = ['src/numc.c', 'src/matrix.c'], 
    	extra_compile_args = CFLAGS, extra_link_args = LDFLAGS)
    setup(name = 'numc', version = '1.0', description = 'This is a numc package.', ext_modules = [module1])

if __name__ == "__main__":
    main()
