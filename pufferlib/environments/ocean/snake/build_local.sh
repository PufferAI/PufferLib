#gcc -g -o snakegame snake.c -I./raylib-5.0_linux_amd64/include/ -L./raylib-5.0_linux_amd64/lib/ -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
clang -fsanitize=address -g -o snakegame snake.c -I./raylib-5.0_linux_amd64/include/ -L./raylib-5.0_linux_amd64/lib/ -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
