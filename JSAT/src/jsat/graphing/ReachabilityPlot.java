
package jsat.graphing;

import java.awt.*;
import jsat.clustering.OPTICS;

/**
 * A reachability plot is a type of visualization that shows the reachability of
 * each data point in a specially sorted order. {@link OPTICS} can generate a 
 * reachability array. 
 * 
 * @author Edward Raff
 */
public class ReachabilityPlot extends Graph2D
{
    private double[] reachability;
    private double max = -Double.MAX_VALUE;
    
    /**
     * Creates a new reachability plot using the given values and order of the 
     * array. It may contain {@link Double#POSITIVE_INFINITY} values. 
     * 
     * @param reachability the sorted order reachability values
     */
    public ReachabilityPlot(double[] reachability)
    {
        super(0, 1, 0, 1);
        this.reachability = reachability;
        for(double d : reachability)
        {
            if(Double.isInfinite(d))
                continue;
            max = Math.max(d, max);
        }
        
        setYMax(max);
        setYAxisTtile("Reachability Distance");
        setXAxisTtile("Reachability Order");
    }

    @Override
    protected void paintWork(Graphics g, int imageWidth, int imageHeight, ProgressPanel pp)
    {
        super.paintWork(g, imageWidth, imageHeight, pp);
        
        g.setColor(Color.BLACK);
        
        int H = imageHeight-getPadding()*2;
        
        int thickness = Math.max((int)Math.floor(imageWidth/(double)reachability.length), 1);
        
        
        for(int i = 0; i < reachability.length; i++)
        {
            int too;
            if(Double.isInfinite(reachability[i]))
                too = H;
            else
                too = (int) (H*(reachability[i]/max));
            g.fillRect(i*thickness+getPadding(), H-too+getPadding(), thickness, too);
        }

    }

    @Override
    public Dimension getPreferredSize()
    {
        return new Dimension(reachability.length+getPadding()*4, 400);
    }

    @Override
    public Dimension getMinimumSize()
    {
        return new Dimension(reachability.length+getPadding()*2+2, 100);
    }
    
    
}
