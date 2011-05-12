
package jsat.math;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class SimpleLinearRegression
{
    /**
     * Performs a Simple Linear Regression on the data set, calculating the best fit a and b such that y = a + b * x <br><br>
     *
     * @param yData the Y data set (to be predicted)
     * @param xData the X data set (the predictor)
     * @return an array containing the a and b, such that index 0 contains a and index 1 contains b
     */
    static public double[] regres(Vec xData, Vec yData)
    {
        //find y = a + B *x
        double[] toReturn = new double[2];

        //B value
        toReturn[1] = DescriptiveStatistics.sampleCorCoeff(xData, yData)*yData.standardDeviation()/xData.standardDeviation();
        //a value
        toReturn[0] = yData.mean() - toReturn[1]*xData.mean();

        return toReturn;
    }
}
