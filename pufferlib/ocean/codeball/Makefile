CFILES=codeball.c
HFILES=codeball.h renderer.h
CFLAGS=\
	-lm -I../../../raylib/include -I../../../pufferlib -lpthread ../../../raylib/lib/libraylib.a \
	-O1 -g -fsanitize=address -fno-omit-frame-pointer \
	-framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo -framework GLUT -framework AppKit -lglfw -lobjc \
	# -O3 -march=native -funroll-loops
SHADERS=base_vs.h fragment_fs.h

ARCH=$(shell uname -m)
PYTHON_VERSION=$(shell python -c 'import sys; print(str(sys.version_info.major) + str(sys.version_info.minor) + "-" + sys.platform)')
PYTHON_SO_POSTFIX=cpython-${PYTHON_VERSION}.so

clean:
	rm -f codeball *.so cy_codeball.c

base_vs.h: base.vs
	xxd -i base.vs > base_vs.h

fragment_fs.h: fragment.fs
	xxd -i fragment.fs > fragment_fs.h

codeball: $(CFILES) $(HFILES) Makefile $(SHADERS)
	gcc $(CFLAGS) $(CFILES) -o codeball

codeball.pxd: cy_codeball.pyx codeball.h
	autopxd codeball.h codeball.pxd

cy_codeball.${PYTHON_SO_POSTFIX}: $(CFILES) $(HFILES) $(SHADERS) cy_codeball.pyx codeball.pxd renderer.pxd setup.py
	python setup.py build_ext --inplace

run_py: cy_codeball.${PYTHON_SO_POSTFIX}
	python -m codeball

run: codeball
	./codeball

.PHONY: clean run run_py