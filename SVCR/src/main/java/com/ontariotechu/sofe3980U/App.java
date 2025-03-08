package com.ontariotechu.sofe3980U;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import java.io.FileReader;
import java.util.List;

/**
 * Class to evaluate binary regression models based on various metrics.
 */
public class App {
    public static void main(String[] args) {
        String[] files = {"model_1.csv", "model_2.csv", "model_3.csv"};
        for (String filePath : files) {
            try (FileReader filereader = new FileReader(filePath);
                 CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build()) {
                List<String[]> allData = csvReader.readAll();

                // Metrics computation
                double bce = calculateBCE(allData);
                int[] confusion = calculateConfusionMatrix(allData, 0.5);
                double accuracy = calculateAccuracy(confusion);
                double precision = calculatePrecision(confusion);
                double recall = calculateRecall(confusion);
                double f1Score = calculateF1Score(precision, recall);

                // Output metrics
                System.out.println("Metrics for " + filePath + ":");
                System.out.println("BCE: " + bce);
                System.out.println("Accuracy: " + accuracy);
                System.out.println("Precision: " + precision);
                System.out.println("Recall: " + recall);
                System.out.println("F1 Score: " + f1Score);
            } catch (Exception e) {
                System.out.println("Error processing the CSV file " + filePath);
                e.printStackTrace();
            }
        }
    }

    /**
     * Calculates the Binary Cross-Entropy (BCE).
     * BCE is used as a loss function for binary classification models.
     */
    public static double calculateBCE(List<String[]> data) {
        double bce = 0;
        for (String[] row : data) {
            double actual = Double.parseDouble(row[0]);
            double predicted = Double.parseDouble(row[1]);
            bce -= (actual * Math.log(predicted + 1e-9) + (1 - actual) * Math.log(1 - predicted + 1e-9));
        }
        return bce / data.size();
    }

    /**
     * Calculates the confusion matrix.
     * Returns an array of int with values [TP, TN, FP, FN].
     */
    public static int[] calculateConfusionMatrix(List<String[]> data, double threshold) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (String[] row : data) {
            double actual = Double.parseDouble(row[0]);
            double predicted = Double.parseDouble(row[1]);
            boolean predLabel = predicted >= threshold;

            if (actual == 1.0) {
                if (predLabel) tp++;
                else fn++;
            } else {
                if (predLabel) fp++;
                else tn++;
            }
        }
        return new int[]{tp, tn, fp, fn};
    }

    /**
     * Calculates Accuracy from confusion matrix.
     */
    public static double calculateAccuracy(int[] confusion) {
        double total = confusion[0] + confusion[1] + confusion[2] + confusion[3];
        return (confusion[0] + confusion[1]) / total;
    }

    /**
     * Calculates Precision from confusion matrix.
     */
    public static double calculatePrecision(int[] confusion) {
        return (double) confusion[0] / (confusion[0] + confusion[2]);
    }

    /**
     * Calculates Recall from confusion matrix.
     */
    public static double calculateRecall(int[] confusion) {
        return (double) confusion[0] / (confusion[0] + confusion[3]);
    }

    /**
     * Calculates F1 Score using Precision and Recall.
     */
    public static double calculateF1Score(double precision, double recall) {
        return 2 * (precision * recall) / (precision + recall);
    }
}
