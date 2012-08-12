
package jsat.graphing;

import java.awt.*;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;

/**
 * A scatter plot to visualize 2 dimensional data sets. 
 * @author Edward Raff
 */
public class ScatterPlot extends Graph2D
{
    private Vec xValues;
    private Vec yValues;
    /**
     * Private function to be used by those who want to plot a regression through the data set
     */
    private Function regressionFunction;

    /**
     * Creates a new scatter plot to visualize the given data
     * @param xValues the x values of the data set
     * @param yValues the y values of the data set
     * @throws ArithmeticException if <tt>xValues</tt> and <tt>yValues</tt> do 
     * not have the same length
     */
    public ScatterPlot(Vec xValues, Vec yValues)
    {
        this(xValues, yValues, null);
    }
    
    /**
     * Creates a new scatter plot to visualize the given data
     * @param xValues the x values of the data set
     * @param yValues the y values of the data set
     * @param regressor the regression method to plot with the data points. If
     * <tt>null</tt>, no regression line will be shown
     * @throws ArithmeticException if <tt>xValues</tt> and <tt>yValues</tt> do 
     * not have the same length
     */
    public ScatterPlot(Vec xValues, Vec yValues, Regressor regressor)
    {
        super(xValues.min(), xValues.max(), yValues.min(), yValues.max());
        if(xValues.length() != yValues.length())
            throw new ArithmeticException("Data sets must have the same length");//TODO more usefull error message
        this.xValues = xValues;
        this.yValues = yValues;
        if(regressor == null)
            regressionFunction = null;
        else
        {
            RegressionDataSet rds = new RegressionDataSet(1, new CategoricalData[0]);
            final int[] catVals = new int[0];
            final CategoricalData[] catData = new CategoricalData[0];
            for(int i = 0; i < xValues.length(); i++)
                rds.addDataPoint(new DataPoint(DenseVector.toDenseVec(xValues.get(i)), catVals, catData), yValues.get(i));
            final Regressor myReg = regressor.clone();
            myReg.train(rds);
            regressionFunction = new Function() {

                @Override
                public double f(double... x)
                {
                    return f(DenseVector.toDenseVec(x));
                }

                @Override
                public double f(Vec x)
                {
                    return myReg.regress(new DataPoint(x, catVals, catData));
                }
            };
        }
    }

    @Override
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        super.paintWork(g, imageWidth, imageHeight, pp);
        Graphics2D g2 = (Graphics2D)g;
        
        if(pp != null)
        {
            pp.getjProgressBar().setIndeterminate(false);
            pp.getjProgressBar().setMaximum(xValues.length());
        }

        g2.setColor(Color.red);
        for(int i = 0; i < xValues.length(); i++ )
        {
            drawPoint(g2, PointShape.CIRCLE, xValues.get(i), yValues.get(i), imageWidth, imageHeight, 6, false);
            if(pp != null)
                pp.getjProgressBar().setValue(i+1);
        }

        if(regressionFunction != null)
        {
            g2.setColor(Color.BLUE);
            drawFunction(g2, regressionFunction);
        }
    }
    
    public void setRegressionFunction(Function regressionFunction)
    {
        this.regressionFunction = regressionFunction;
    }

    public Function getRegressionFunction()
    {
        return regressionFunction;
    }

}
