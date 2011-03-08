
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
public class ScatterPlot extends Graph2D
{
    private Vec xValues;
    private Vec yValues;
    /**
     * Private function to be used by those who want to plot a regression through the data set
     */
    private Function regressionFunction;

    public ScatterPlot(Vec xValues, Vec yValues)
    {
        super(xValues.min()-2, xValues.max()+2, yValues.min()-2, yValues.max()+2);
        this.xValues = xValues;
        this.yValues = yValues;
        regressionFunction = null;
    }

    @Override
    protected void paintComponent(Graphics g)
    {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D)g;

        g2.setColor(Color.red);
        for(int i = 0; i < xValues.length(); i++ )
        {
            g2.draw(new Ellipse2D.Double(toXCord(xValues.get(i))-3, toYCord(yValues.get(i))-3, 6, 6));
        }

        if(regressionFunction != null)
            drawFunction(regressionFunction, g2, Color.BLUE);
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
