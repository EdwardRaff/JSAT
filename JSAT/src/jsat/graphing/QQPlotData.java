
package jsat.graphing;

import java.awt.*;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 * Performs a QQ plot to compare the distribution of two sets of univariate data
 * that have the same number of data points. 
 * 
 * @author Edward Raff
 */
public class QQPlotData extends Graph2D
{
    private Vec yData;
    private Vec xData;

    /**
     * Creates a new QQ plot using the given data
     * @param xData the first set of data
     * @param yData the second set of data
     * @throws ArithmeticException if the data sets are not of the same size
     */
    public QQPlotData(Vec xData, Vec yData)
    {
        super(0, 0, 0, 0);
        if(xData.length() != yData.length())
            throw new ArithmeticException("Data sets must be the same size");
        this.yData = yData.sortedCopy();
        this.xData = xData.sortedCopy();

        double min = Math.min(this.yData.min(), this.xData.min());
        double max = Math.max(this.yData.max(), this.xData.max());

        setXMin(min);
        setXMax(max);
        setYMin(min);
        setYMax(max);
    }

    @Override
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        super.paintWork(g, imageWidth, imageHeight, pp);
        Graphics2D g2 = (Graphics2D)g;

        g2.setColor(Color.red);
        for(int i = 0; i < yData.length(); i++ )
        {
            drawPoint(g2, PointShape.CIRCLE, xData.get(i), yData.get(i), imageWidth, imageHeight, 6, false);
        }

        g.setColor(Color.BLUE);
        drawFunction(g2, new Function()
        {

            @Override
            public double f(double... x)
            {
                return x[0];
            }

            @Override
            public double f(Vec x)
            {
                return x.get(0);
            }
        });
    }
}
