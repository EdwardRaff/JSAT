
package jsat.graphing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import jsat.distributions.ContinousDistribution;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class QQPlotDistribution extends Graph2D
{
    ContinousDistribution cd;
    private Vec yData;
    private ArrayList<Double> xData;

    public QQPlotDistribution(ContinousDistribution cd, Vec data)
    {
        super(0, 0, 0, 0);
        this.cd = cd;
        this.yData = data.sortedCopy();

        xData = new ArrayList<Double>(yData.length());

        double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        for(int i = 0; i < yData.length(); i++ )
        {
            double v = (i+1.0-0.375)/(yData.length()+0.25);
            double x = cd.invCdf(v);
            min = Math.min(x, min);
            max = Math.max(x, max);

            xData.add(x);
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
    protected void paintComponent(Graphics g)
    {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D)g;

        g2.setColor(Color.red);
        for(int i = 0; i < yData.length(); i++ )
        {

            g2.draw(new Ellipse2D.Double(toXCord(xData.get(i))-3, toYCord(yData.get(i))-3, 6, 6));
        }

        drawFunction(new Function() {

            public double f(double... x)
            {
                return x[0];
            }

            public double f(Vec x)
            {
                return x.get(0);
            }
        }, g2, Color.BLUE);
    }





}
