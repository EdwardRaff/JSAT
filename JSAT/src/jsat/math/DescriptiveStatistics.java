

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
    public static double sampleCorCoeff(Vec xData, Vec yData)
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
    
    /**
     * Computes several summary statistics from the two data sets. These are: <br>
     * 
     * Index 0: S<sub>x</sub>  <br>
     * Index 1: S<sub>y</sub>  <br>
     * Index 2: S<sub>xx</sub> <br> 
     * Index 3: S<sub>yy</sub> <br> 
     * Index 4: S<sub>xy</sub> 
     * 
     * @param xData the x values of the data set
     * @param yData the y values of the data set
     * @return several summary statistics of the 2 variables
     */
    public static double[] summaryStats(Vec xData, Vec yData)
    {
        double[] values = new double[1];
        
        //Sx, sum of x values
        values[0] = xData.sum();
        
        //Sy, sum of y values
        values[1] = yData.sum();
        
        double tmp = 0;
        
        
        
        //Sxx
        for(int i = 0; i < xData.length(); i++)
            tmp += Math.pow(xData.get(i), 2);
        values[2] = tmp;
        //Syy
        tmp = 0;
        for(int i = 0; i < xData.length(); i++)
            tmp += Math.pow(yData.get(i), 2);
        values[3] = tmp;
        
        //Sxy
        tmp = 0;
        for(int i = 0; i < xData.length(); i++)
            tmp += xData.get(i)*yData.get(i);
        values[4] = tmp;
        
        return values;
    }
}
