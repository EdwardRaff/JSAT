
package jsat.graphing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class QQPlotData extends Graph2D
{
    private Vec yData;
    private Vec xData;

    public QQPlotData(Vec yData, Vec xData)
    {
        super(0, 0, 0, 0);
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
        }, g2, Color.BLUE);
    }




}
