

package jsat.graphing;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.text.DecimalFormat;
import javax.swing.JComponent;
import jsat.distributions.Normal;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class Graph2D extends JComponent
{
    protected double xMin;
    protected double xMax;

    protected double yMin;
    protected double yMax;

    protected int PAD;
    protected String xAxisTtile = "X Axis";
    protected String yAxisTtile = "Y Axis";

    public Graph2D(double xMin, double xMax, double yMin, double yMax)
    {
        this.xMin = xMin;
        this.xMax = xMax;
        this.yMin = yMin;
        this.yMax = yMax;
        this.PAD = -1;//AUTO SET FOR ME PLEASE
    }

    public void setYAxisTtile(String yAxisTtile)
    {
        this.yAxisTtile = yAxisTtile;
    }

    public void setXAxisTtile(String xAxisTtile)
    {
        this.xAxisTtile = xAxisTtile;
    }

    public void setXMin(double xMin)
    {
        this.xMin = xMin;
    }

    public void setXMax(double xMax)
    {
        this.xMax = xMax;
    }

    public void setYMin(double yMin)
    {
        this.yMin = yMin;
    }

    public void setYMax(double yMax)
    {
        this.yMax = yMax;
    }

    public void setPadding(int p)
    {
        this.PAD = p;
    }

    public int getPadding()
    {
        return PAD;
    }
    
    @Override
    protected void paintComponent(Graphics g)
    {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D)g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                            RenderingHints.VALUE_ANTIALIAS_ON);
        int w = getWidth();
        int h = getHeight();

        FontMetrics fm = g2.getFontMetrics();

        if(PAD == 0)//No padding, dont draw borders!
            return;
        else if(PAD == -1)//auto set
            PAD = (fm.getHeight()+2)*2;//Auto pad so we have 2 levels

        g2.setColor(Color.black);
         // Draw horizontal.
        g2.draw(new Line2D.Double(PAD, PAD, PAD, h-PAD));
        // Draw vertical.
        g2.draw(new Line2D.Double(PAD, h-PAD, w-PAD, h-PAD));

        
        
        //Draw horizontal marks
        for(int x = PAD*2; x <= w-PAD*2; x+= (w-PAD*2)/4)
        {
            double xVal = toXVal(x);
            String xValString = displayValue(xVal);
            
            g2.drawString(xValString, x-fm.stringWidth(xValString)/2, h-PAD+fm.getHeight());
        }
        
        //draw X axist title
        if(xAxisTtile != null && !xAxisTtile.equals(""))
            g2.drawString(xAxisTtile, (w-PAD)/2-fm.stringWidth(xAxisTtile)/2, h-PAD+fm.getHeight()*2+2);

        //Dray Y axis title verticaly
        AffineTransform at = new AffineTransform();
        at.setToRotation(-Math.PI/2.0, 0, 0);

        Font origFont = getFont();
        Font der = origFont.deriveFont(at);
        g2.setFont(der);

        //Draw vertical marks
        for(int y = PAD*2; y <= h-PAD*2; y+= (h-PAD*2)/4)
        {
            double yVal = toYVal(y);
            String yValString = displayValue(yVal);
            
            g2.drawString(yValString, fm.getHeight()*2, y+fm.getHeight()/2);
        }
        
        if(yAxisTtile != null && !yAxisTtile.equals(""))
            g2.drawString(yAxisTtile, fm.getAscent(), (h+PAD)/2+fm.getHeight());
        
        g2.setFont(origFont);

    }
    
    
    private static final DecimalFormat sigFormat = new DecimalFormat("0.##E0");
    private static String displayValue(double x)
    {
        String tmp = Double.toString(x);
        if(tmp.length() <= 4)
            return tmp;
        else if( x < 99 && x >= -99)
            return tmp.substring(0, 5);
        else if (x > 0 && x < 9999)//If id did not have a decimal poitn in it, we would have returned in the first if
            return tmp.substring(0, 5);
        else if(x < 0 && x > -9999 && tmp.length() > 5)//Same logic
            return tmp.substring(0, 6);
        //Else, sig fig it
        
        return sigFormat.format(x);
    }

    public int toXCord(double x)
    {
        double scale = (getWidth() - 2*PAD)/(xMax-xMin);

        x-=xMin;

        return (int)Math.round(x*scale)+PAD;
    }

    public double toXVal(int x)
    {
        x-=PAD;
        double scale = (getWidth() - 2*PAD)/(xMax-xMin);

        return x/scale+xMin;
    }
    
    public double toYVal(int y)
    {
        y-=PAD;
        y = (getHeight()-2*PAD) - y;//Becase Java swing, the y axis is 0 at the top and H at the bottom, oposite of what we want
        double scale = (getHeight() - 2*PAD)/(yMax-yMin);

        return y/scale+yMin;
    }

    public int toYCord(double y)
    {
        double scale = (getHeight() - 2*PAD)/(yMax-yMin);

        y-=yMin;

        return getHeight() - (int)Math.round(y*scale)-PAD;
    }

    protected void drawFunction(Function func, Graphics2D g, Color c)
    {
        double lastX = toXVal(PAD+1);
        double lastY = func.f(lastX);
        g.setColor(c);

        for(int i = PAD+2; i < getWidth()-PAD; i+=4)
        {
            double x = toXVal(i);
            double y = func.f(x);
            
            g.drawLine(toXCord(lastX), toYCord(lastY), toXCord(x), toYCord(y));
            lastX = x;
            lastY = y;
        }
    }


}
