
package jsat.graphing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import jsat.distributions.Distribution;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 * Creates a new QQ plot to compare a data set against a specified distribution
 * @author Edward Raff
 */
public class QQPlotDistribution extends Graph2D
{
    private Distribution cd;
    private Vec yData;
    private double[] xData;

    /**
     * Creates a new QQ plot to compare the given data set with a known distribution
     * @param cd the distribution to compare against. 
     * @param data the data set to compare
     */
    public QQPlotDistribution(Distribution cd, Vec data)
    {
        super(0, 0, 0, 0);
        this.cd = cd;
        this.yData = data.sortedCopy();

        xData = new double[yData.length()];

        double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        for(int i = 0; i < yData.length(); i++ )
        {
            double v = (i+1.0-0.375)/(yData.length()+0.25);
            double x = cd.invCdf(v);
            min = Math.min(x, min);
            max = Math.max(x, max);

            xData[i]=x;
        }

        min = Math.min(yData.get(0), min);
        max = Math.max(yData.max(), max);

        setXMin(min);
        setXMax(max);
        setYMin(min);
        setYMax(max);

        setXAxisTtile(cd.getDescriptiveName());

    }

    @Override
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        super.paintWork(g, imageWidth, imageHeight, pp);
        Graphics2D g2 = (Graphics2D)g;

        g2.setColor(Color.red);
        for(int i = 0; i < yData.length(); i++ )
            drawPoint(g2, PointShape.CIRCLE, xData[i], yData.get(i), imageWidth, imageHeight, 6, false);

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
