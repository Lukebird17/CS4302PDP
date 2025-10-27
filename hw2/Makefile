# 定义编译器
CC = nvcc

# 定义编译选项
CFLAGS = -O3

# 源文件
SRC1 = prob1.cu
SRC2 = prob2.cu

# 目标文件
OBJ1 = prob1.o
OBJ2 = prob2.o

# 最终生成的可执行文件
EXEC1 = prob1.out
EXEC2 = prob2.out

# 默认规则：生成可执行文件
$(EXEC1): $(OBJ1)
	$(CC) $(CFLAGS) -o $(EXEC1) $(OBJ1)

$(EXEC2): $(OBJ2)
	$(CC) $(CFLAGS) -o $(EXEC2) $(OBJ2)

# 编译规则：将 .cu 文件编译为 .o 文件
$(OBJ1): $(SRC1)
	$(CC) $(CFLAGS) -c $(SRC1)

$(OBJ2): $(SRC2)
	$(CC) $(CFLAGS) -c $(SRC2)

# 清理规则：删除生成的 .o 和可执行文件
clean:
	rm -f $(OBJ1) $(OBJ2) $(EXEC1) $(EXEC2)
