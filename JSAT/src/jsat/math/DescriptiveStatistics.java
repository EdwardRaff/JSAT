

package jsat.math;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class DescriptiveStatistics
{
    /**
     * Computes the sample correlation coefficient for two data sets X and Y. The lengths of X and Y must be the same, and each element in X should correspond to the element in Y.
     *
     * @param yData the Y data set
     * @param xData the X data set
     * @return the sample correlation coefficient
     */
    public static double sampleCorCoeff(Vec yData, Vec xData)
    {
        if(yData.length() != xData.length())
            throw new ArithmeticException("X and Y data sets must have the same length");

        double xMean = xData.mean();
        double yMean = yData.mean();

        double topSum = 0;

        for(int i = 0; i < xData.length(); i++)
        {
            topSum += (xData.get(i)-xMean)*(yData.get(i)-yMean);
        }


        return topSum/((xData.length()-1)*xData.standardDeviation()*yData.standardDeviation());
    }
}
