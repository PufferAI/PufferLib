clang -Wall -Wuninitialized -Wmisleading-indentation -fsanitize=address,undefined,bounds,pointer-overflow,leak -ferror-limit=3 -g -o robocode robocode.c -I./raylib-5.0_linux_amd64/include/ -L./raylib-5.0_linux_amd64/lib/ -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -DPLATFORM_DESKTOP


