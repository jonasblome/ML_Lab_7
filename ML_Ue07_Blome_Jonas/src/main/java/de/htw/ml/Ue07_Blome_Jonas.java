package de.htw.ml;

import java.awt.*;
import java.io.IOException;
import java.util.Arrays;

import org.jblas.FloatMatrix;
import org.jblas.util.Random;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class Ue07_Blome_Jonas {
    public static final String title = "Credit approximation";
    public static final String xAxisLabel = "Iteration count";
    public static final String yAxisLabel = "Prediction percentage";

    public static void main(String[] args) throws IOException {
        // THERE IS NO PLOTTING IN MY SOLUTION BECAUSE I COULDN'T GET JAVA FX TO WORK PROPERLY!!!!!
        // RATHER THE DEVELOPMENT OF THE PREDICTION RATE IS VISIBLE IN THE CONSOLE AS A PRINTED ARRAY
        System.out.println("Predicting with linear regression");
        approximateCreditsGiven();

        System.out.println();

        System.out.println("Predicting with logistic regression");
        approximateCreditsGivenSigmoid();
    }

    private static void approximateCreditsGiven() throws IOException {
        FloatMatrix germanCredit = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
        int numDatapoints = germanCredit.rows;
        int numIterations = 1000;

        // Alpha values
        float[] learningRates = {0.001f, 0.005f, 0.01f, 0.1f};

        // RMSE array for each iteration
        float[][] rmseValues = new float[learningRates.length][numIterations];

        // Store max and min values of credit columns for normalization
        float[] maxValues = new float[germanCredit.columns];

        for(int col = 0; col < germanCredit.columns; col++) {
            maxValues[col] = germanCredit.getColumn(col).max();
        }

        float[] minValues = new float[germanCredit.columns];

        for(int col = 0; col < germanCredit.columns; col++) {
            minValues[col] = germanCredit.getColumn(col).min();
        }

        // Normalize credit data
        float[][] normalizedData = new float[germanCredit.columns][germanCredit.rows];

        for(int col = 0; col < germanCredit.columns; col++) {
            normalizedData[col] = germanCredit.getColumn(col).sub(minValues[col]).div(maxValues[col] - minValues[col]).toArray();
        }

        // Turn normalized data arrays into vectors
        FloatMatrix[] normalizedVectors = new FloatMatrix[germanCredit.columns];

        for(int col = 0; col < germanCredit.columns; col++) {
            normalizedVectors[col] = new FloatMatrix(normalizedData[col]);
        }

        // Put normalized credit data into matrix
        FloatMatrix inputMatrix = new FloatMatrix(normalizedData[1]);

        for(int col = 2; col < germanCredit.columns; col++) {
            inputMatrix = FloatMatrix.concatHorizontally(inputMatrix, normalizedVectors[col]);
        }

        float[][] predictionRate = new float[learningRates.length][numIterations];

        // Create four different learning curves
        for(int lc = 0; lc < learningRates.length; lc++) {
            Random.seed(7);

            // Set initial theta values
            float bestRMSE = 100000;
            float[] thetaValues = new float[germanCredit.columns - 1];
            int thetaMin = -1;
            int thetaMax = 1;
            for(int theta = 0; theta < thetaValues.length; theta++) {
                thetaValues[theta] = thetaMin + Random.nextFloat() * (thetaMax - thetaMin);
            }

            // Turn theta values into vector
            FloatMatrix thetaValuesVector = new FloatMatrix(thetaValues);

            // Try multiple iterations to find the best theta values
            for(int it = 0; it < numIterations; it++) {
                // Calculate approximated credit amount values with weighted function (hypothesis function)
                FloatMatrix hypothesis = inputMatrix.mmul(thetaValuesVector);

                // Calculate disparity
                FloatMatrix disparity = hypothesis.sub(normalizedVectors[0]);

                // Calculate theta-delta values and update them
                FloatMatrix thetaDeltaVector = inputMatrix.transpose().mmul(disparity);
                FloatMatrix normalizedThetaDeltaVector = thetaDeltaVector.mul(learningRates[lc] / numDatapoints);
                thetaValuesVector = thetaValuesVector.sub(normalizedThetaDeltaVector);

                // Calculate RMSE
                float squareErrorSum = 0;

                for(int row = 0; row < numDatapoints; row++) {
                    squareErrorSum += Math.pow(((maxValues[0] - minValues[0]) * hypothesis.get(row) + minValues[0]) - ((maxValues[0] - minValues[0]) * normalizedData[0][row]) + minValues[0], 2);
                }

                float rmse = (float) Math.sqrt(squareErrorSum / numDatapoints);

                // Store current rmse & update the best thetas if rmse is smaller
                rmseValues[lc][it] = rmse;

                if(rmse < bestRMSE) {
                    bestRMSE = rmse;
                }

                float[] hypothesisBin = new float[numDatapoints];

                // Binarize hypothesis output to derive prediction rate
                for(int h = 0; h < numDatapoints; ++h) {
                    if (hypothesis.get(h) >= 0.5) {
                        hypothesisBin[h] = 1;
                    }
                    else{
                        hypothesisBin[h] = 0;
                    }
                }

                FloatMatrix predictionErrorVector = new FloatMatrix(hypothesisBin).sub(normalizedVectors[0]);

                float[] predictionErrorArray = predictionErrorVector.toArray();
                int predictionError = 0;

                for(float v: predictionErrorArray) {
                    predictionError += Math.abs(v);
                }

                predictionRate[lc][it] = (float) (numDatapoints - predictionError) / numDatapoints * 100.0f;
            }
            System.out.println("RMSE for creditsGiven learning curve " + lc +": " + bestRMSE);
            System.out.println("Prediction rate per generation: " + Arrays.toString(predictionRate[lc]));
        }
        // Plot
        // FXApplication.plot(predictionRate[3]);
        // Application.launch(FXApplication.class);
    }

    private static void approximateCreditsGivenSigmoid() throws IOException {
        FloatMatrix germanCredit = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
        int numDatapoints = germanCredit.rows;
        int numIterations = 1000;
        float testSetSize = 0.1f;

        // Alpha values
        float[] learningRates = {0.01f, 0.1f, 0.5f, 1.0f};

        // RMSE array for each iteration
        float[][] rmseValues = new float[learningRates.length][numIterations];

        // Store max and min values of credit columns for normalization
        float[] maxValues = new float[germanCredit.columns];

        for(int col = 0; col < maxValues.length; col++) {
            maxValues[col] = germanCredit.getColumn(col).max();
        }

        float[] minValues = new float[germanCredit.columns];

        for(int col = 0; col < minValues.length; col++) {
            minValues[col] = germanCredit.getColumn(col).min();
        }

        // Normalize credit data
        float[][] normalizedData = new float[germanCredit.columns][germanCredit.rows];

        for(int col = 0; col < maxValues.length; col++) {
            normalizedData[col] = germanCredit.getColumn(col).sub(minValues[col]).div(maxValues[col] - minValues[col]).toArray();
        }

        // Turn normalized data arrays into vectors
        FloatMatrix[] normalizedVectors = new FloatMatrix[germanCredit.columns];

        for(int col = 0; col < normalizedVectors.length; col++) {
            normalizedVectors[col] = new FloatMatrix(normalizedData[col]);
        }

        // Put credit data into matrix
        FloatMatrix inputMatrix = new FloatMatrix(normalizedData[1]);

        for(int i = 2; i < normalizedVectors.length; i++) {
            inputMatrix = FloatMatrix.concatHorizontally(inputMatrix, normalizedVectors[i]);
        }

        float [][] predictionRate = new float[learningRates.length][numIterations];

        // Split input data into training and test set
        FloatMatrix testSet = FloatMatrix.zeros((int) (numDatapoints * testSetSize), germanCredit.columns - 1);
        FloatMatrix originalSetTest = FloatMatrix.zeros((int) (numDatapoints * testSetSize), 1);
        FloatMatrix trainSet = FloatMatrix.zeros((int) (numDatapoints * (1 - testSetSize)), germanCredit.columns - 1);

        int positivesInTest = 0;
        int negativesInTest = 0;

        for(int row = 0; row < numDatapoints; row++) {
            if(germanCredit.get(row, 0) == 1 && positivesInTest < numDatapoints * testSetSize / 2) {
                testSet.putRow(positivesInTest + negativesInTest, inputMatrix.getRow(row));
                originalSetTest.put(positivesInTest + negativesInTest, 0, germanCredit.getColumn(0).toArray()[row]);
                positivesInTest++;
            }
            else if(germanCredit.get(row, 0) == 0 && negativesInTest < numDatapoints * testSetSize / 2) {
                testSet.putRow(positivesInTest + negativesInTest, inputMatrix.getRow(row));
                originalSetTest.put(positivesInTest + negativesInTest, 0, germanCredit.getColumn(0).toArray()[row]);
                negativesInTest++;
            }
            else{
                trainSet.putRow(row - positivesInTest - negativesInTest, inputMatrix.getRow(row));
            }
        }

        // Create four different learning curves
        for(int lc = 0; lc < learningRates.length; lc++) {
            Random.seed(7);

            // Set initial theta values
            float bestRMSE = 100000;
            float[] thetaValues = new float[germanCredit.columns - 1];
            int thetaMin = -1;
            int thetaMax = 1;
            for(int theta = 0; theta < thetaValues.length; theta++) {
                thetaValues[theta] = thetaMin + Random.nextFloat() * (thetaMax - thetaMin);
            }

            // Turn theta values into vector
            FloatMatrix thetaValuesVector = new FloatMatrix(thetaValues);

            // Try multiple iterations to find the best theta values
            for(int it = 0; it < numIterations; it++) {
                // Calculate approximated credit amount values with weighted function (hypothesis function)
                FloatMatrix hypothesisTrain = trainSet.mmul(thetaValuesVector);
                FloatMatrix hypothesisTest = testSet.mmul(thetaValuesVector);
                float[] hypothesisTrainValues = hypothesisTrain.toArray();
                float[] hypothesisTestValues = hypothesisTest.toArray();

                // Squeeze Hypothesis into Sigmoid function
                float[] sigmoidTrain = new float[hypothesisTrainValues.length];
                float[] sigmoidTest = new float[hypothesisTestValues.length];

                for(int h = 0; h < hypothesisTrainValues.length; h++) {
                    sigmoidTrain[h] = (float) (1 / (1 + Math.pow(Math.E, -hypothesisTrainValues[h])));
                }

                for(int row = 0; row < hypothesisTestValues.length; row++) {
                    sigmoidTest[row] = (float) (1 / (1 + Math.pow(Math.E, -hypothesisTestValues[row])));
                }

                // Calculate disparity
                float[] disparity = new float[numDatapoints];

                for(int row = 0; row < sigmoidTrain.length; row++) {
                    disparity[row] = sigmoidTrain[row] - normalizedData[0][row];
                }

                FloatMatrix disparityVector = new FloatMatrix(disparity);

                // Calculate theta-delta values and update them
                FloatMatrix thetaDeltaVector = inputMatrix.transpose().mmul(disparityVector);
                FloatMatrix normalizedThetaDeltaVector = thetaDeltaVector.mul(learningRates[lc] / numDatapoints);
                thetaValuesVector = thetaValuesVector.sub(normalizedThetaDeltaVector);

                // Calculate RMSE
                float squareErrorSum = 0;

                for(int row = 0; row < hypothesisTrainValues.length; row++) {
                    squareErrorSum += Math.pow(((maxValues[0] - minValues[0]) * sigmoidTrain[row] + minValues[0]) - ((maxValues[0] - minValues[0]) * normalizedData[0][row]) + minValues[0], 2);
                }

                float rmse = (float) Math.sqrt(squareErrorSum / numDatapoints);

                // Store current rmse & update the best thetas if rmse is smaller
                rmseValues[lc][it] = rmse;

                if(rmse < bestRMSE) {
                    bestRMSE = rmse;
                }

                // Binarize hypothesis output to derive prediction rate
                float[] testBin = new float[sigmoidTest.length];

                for(int row = 0; row < sigmoidTest.length; row++) {
                    if (sigmoidTest[row] >= 0.5) {
                        testBin[row] = 1;
                    }
                    else{
                        testBin[row] = 0;
                    }
                }

                FloatMatrix predictionErrorVector = new FloatMatrix(testBin).sub(originalSetTest);

                float[] predictionErrorArray = predictionErrorVector.toArray();
                int predictionError = 0;

                for(float err: predictionErrorArray) {
                    predictionError += Math.abs(err);
                }

                predictionRate[lc][it] = (float) (numDatapoints - predictionError) / numDatapoints * 100.0f;
            }

            System.out.println("RMSE for creditsGiven learning curve " + lc +": " + bestRMSE);
            System.out.println("Prediction rate per generation: " + Arrays.toString(predictionRate[lc]));
        }
        // Plot
        // FXApplication.plot(predictionRate[3]);
        // Application.launch(FXApplication.class);float[]
    }


    // ---------------------------------------------------------------------------------
    // ------------ Alle Änderungen ab hier geschehen auf eigene Gefahr ----------------
    // ---------------------------------------------------------------------------------

    /**
     * We need a separate class in order to trick Java 11 to start our JavaFX application without any module-path settings.
     * https://stackoverflow.com/questions/52144931/how-to-add-javafx-runtime-to-eclipse-in-java-11/55300492#55300492
     *
     * @author Nico Hezel
     *
     */
    public static class FXApplication extends Application {

        /**
         * equivalent to linspace in Octave
         *
         * @param lower
         * @param upper
         * @param num
         * @return
         */
        private static FloatMatrix linspace(float lower, float upper, int num) {
            float[] data = new float[num];
            float step = Math.abs(lower-upper) / (num-1);
            for (int i = 0; i < num; i++)
                data[i] = lower + (step * i);
            data[0] = lower;
            data[data.length-1] = upper;
            return new FloatMatrix(data);
        }

        // y-axis values of the plot
        private static float[] dataY;

        /**
         * Draw the values and start the UI
         */
        public static void plot(float[] yValues) {
            dataY = yValues;
        }

        /**
         * Draw the UI
         */
        @SuppressWarnings("unchecked")
        @Override
        public void start(Stage stage) {

            stage.setTitle(title);

            final NumberAxis xAxis = new NumberAxis();
            xAxis.setLabel(xAxisLabel);
            final NumberAxis yAxis = new NumberAxis();
            yAxis.setLabel(yAxisLabel);

            final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);

            XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
            series1.setName("Data");
            for (int i = 0; i < dataY.length; i++) {
                series1.getData().add(new XYChart.Data<Number, Number>(i, dataY[i]));
            }

            sc.setAnimated(false);
            sc.setCreateSymbols(true);

            sc.getData().addAll(series1);

            Scene scene = new Scene(sc, 500, 400);
            stage.setScene(scene);
            stage.show();
        }
    }
}
