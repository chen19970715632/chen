package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
)

func loadDataset(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var dataset [][]float64
	for _, row := range rows {
		var sample []float64
		for _, val := range row {
			sample = append(sample, parseFloat(val))
		}
		dataset = append(dataset, sample)
	}

	return dataset, nil
}

func splitDataset(dataset [][]float64, splitRatio float64) ([][]float64, [][]float64) {
	var train [][]float64
	var test [][]float64

	for i := range dataset {
		if rand.Float64() < splitRatio {
			train = append(train, dataset[i])
		} else {
			test = append(test, dataset[i])
		}
	}

	return train, test
}

func saveDataset(filename string, data [][]float64) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	for _, row := range data {
		var strRow []string
		for i := 0; i < 6 && i < len(row); i++ { // 只保存前八列数据
			strRow = append(strRow, fmt.Sprintf("%f", row[i]))
		}
		writer.Write(strRow)
	}

	writer.Flush()

	return nil
}

func parseFloat(value string) float64 {
	var resultVal float64
	_, err := fmt.Sscanf(value, "%f", &resultVal)
	if err != nil {
		return 0
	}
	return resultVal
}

func main() {
	datasetFilename := "./test/data.csv"
	trainFilename := "./test/train_datasetA.csv"
	testFilename := "./test/test_datasetA.csv"
	splitRatio := 0.7 // 训练集比例

	dataset, err := loadDataset(datasetFilename)
	if err != nil {
		panic(err)
	}

	train, test := splitDataset(dataset, splitRatio)

	err = saveDataset(trainFilename, train)
	if err != nil {
		panic(err)
	}

	err = saveDataset(testFilename, test)
	if err != nil {
		panic(err)
	}

	fmt.Printf("数据集已拆分为训练集和测试集并保存到 %s 和 %s 中", trainFilename, testFilename)
}
