package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	inputFile := "./test/housing.data"
	outputFile := "./test/data.csv"

	// 打开输入文件
	file, err := os.Open(inputFile)
	if err != nil {
		fmt.Println("Failed to open input file:", err)
		return
	}
	defer file.Close()

	// 创建CSV输出文件
	csvFile, err := os.Create(outputFile)
	if err != nil {
		fmt.Println("Failed to create output file:", err)
		return
	}
	defer csvFile.Close()

	// 逐行读取输入文件，并将其转换为CSV格式写入输出文件
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)       // 拆分行中的单词
		csvLine := strings.Join(fields, ",") // 将单词使用逗号组合起来
		fmt.Fprintln(csvFile, csvLine)       // 写入CSV文件
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading input file:", err)
		return
	}

	fmt.Println("Data file converted to CSV successfully!")
}
